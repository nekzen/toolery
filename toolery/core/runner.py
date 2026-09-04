from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from toolery.adapters.base import Adapter
from toolery.core.models import Scenario, ScenarioResult
from toolery.core.scorer import evaluate

ResultCallback = Callable[[ScenarioResult], Awaitable[None] | None]
StartCallback = Callable[[str, str, int, str], Awaitable[None] | None]
EndCallback = Callable[[str, str, int], Awaitable[None] | None]

_log = logging.getLogger(__name__)

# Sentinel returned by a unit that was gated off by should_stop() before it
# started. Distinct from None (a valid-ish callback return) so the gather loop
# can tell "skipped, never ran" apart from a real ScenarioResult.
_SKIP = object()

# Substrings that identify a TRANSIENT adapter failure — rate limiting,
# timeouts, connection resets, and 5xx server errors. These are infra hiccups,
# not genuine model failures (bad tool call, wrong answer, malformed output),
# and are safe to retry without inflating a model's apparent capability.
# Deliberately conservative: anything not matched here is treated as a real
# failure and is never retried.
_TRANSIENT_ERROR_MARKERS = (
    "429", "rate limit", "too many requests",
    "timeout", "timed out",
    "connection reset", "connection refused", "connection aborted",
    "connection error", "broken pipe",
    "server error", "bad gateway", "service unavailable", "gateway timeout",
    "500", "502", "503", "504",
)


def is_transient_error(error: str) -> bool:
    """True if `error` looks like a transient adapter failure worth retrying.

    NEVER matches genuine model failures — e.g. "unknown tool", schema
    validation failures, or any error string that doesn't mention rate
    limiting / timeouts / connection issues / 5xx."""
    e = error.lower()
    return any(marker in e for marker in _TRANSIENT_ERROR_MARKERS)


async def _maybe_call(cb, *args) -> None:
    if cb is None:
        return
    try:
        out = cb(*args)
        if asyncio.iscoroutine(out):
            await out
    except Exception:
        _log.exception("callback %s failed", getattr(cb, "__name__", cb))


@dataclass
class Runner:
    adapters: dict[str, Adapter]
    trials: int = 1
    model: str = "model"
    concurrency: int = 4
    skip: set[tuple[str, str, int]] | None = None
    on_start: StartCallback | None = None
    on_end: EndCallback | None = None
    # Multiplier on each scenario's timeout_seconds. The per-scenario budgets
    # were tuned for fast local serving at low concurrency; the MiMo-V2.5 audit
    # (2026-06-12) showed reasoning models killed mid-first-answer at scale 1.0
    # (37 trials, zero messages in trace, passes of the same scenarios landing
    # at 96-100% of the limit). Default is 2.0 since then; bump higher for
    # slow cloud/reasoning endpoints (e.g. 4.0).
    timeout_scale: float = 2.0
    # Retry policy for transient adapter failures (429 / timeout / connection
    # reset / 5xx). Genuine model failures (bad tool call, wrong answer, a
    # clean non-transient error) are NEVER retried — retrying those would
    # silently inflate a model's apparent reliability. max_retries=0 disables
    # retry entirely (single attempt, current behavior).
    max_retries: int = 0
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 30.0

    async def _run_one(self, scenario: Scenario, adapter_name: str, adapter: Adapter,
                       trial_index: int) -> ScenarioResult:
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        await _maybe_call(self.on_start, scenario.id, adapter_name, trial_index, started_at)
        eff_timeout = int(scenario.budget.timeout_seconds * self.timeout_scale)
        try:
            attempt = 0
            while True:
                try:
                    trace = await asyncio.wait_for(
                        adapter.run_scenario(scenario, self.model, eff_timeout),
                        timeout=eff_timeout + 5,
                    )
                except TimeoutError:
                    from toolery.core.models import TraceResult
                    trace = TraceResult(
                        scenario_id=scenario.id, adapter=adapter_name, trial_index=trial_index,
                        messages=[], tool_calls=[], final_response=None,
                        started_at_iso=started_at,
                        duration_ms=eff_timeout * 1000,
                        error="timeout",
                    )
                if attempt < self.max_retries and trace.error and is_transient_error(trace.error):
                    delay = min(self.retry_backoff_base * (2 ** attempt), self.retry_backoff_max)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                break
            trace = trace.model_copy(update={"adapter": adapter_name, "trial_index": trial_index})
            return evaluate(scenario, trace)
        finally:
            await _maybe_call(self.on_end, scenario.id, adapter_name, trial_index)

    async def run(self, scenarios: Iterable[Scenario],
                  on_result: ResultCallback | None = None,
                  should_stop: Callable[[], bool] | None = None,
                  ) -> list[ScenarioResult]:
        """Execute all (scenario × adapter × trial) units concurrently.

        If on_result is provided, it is invoked for each ScenarioResult as soon as
        the unit completes (in completion order, not submission order). The
        callback may be sync or async. Exceptions in the callback are swallowed
        so a single bad observer cannot abort the whole run.

        If should_stop is provided, it is consulted after a unit acquires its
        concurrency slot but before it starts running. Once it returns True, no
        further units begin — yet units already in flight finish normally and
        record their results. This is the graceful-pause path: it drains rather
        than cancels, so a paused run can resume from the next not-yet-run unit.
        """
        sem = asyncio.Semaphore(self.concurrency)

        async def bounded(coro):
            async with sem:
                if should_stop is not None and should_stop():
                    # Stop scheduling new work: discard the not-yet-started
                    # coroutine (so on_start never fires / no in_flight row)
                    # and signal skip. In-flight units past this gate keep
                    # running to completion.
                    coro.close()
                    return _SKIP
                return await coro

        skip = self.skip or set()
        tasks: list[asyncio.Task[ScenarioResult]] = []
        for s in scenarios:
            for adapter_name, adapter in self.adapters.items():
                for t in range(self.trials):
                    if (s.id, adapter_name, t) in skip:
                        continue
                    tasks.append(asyncio.create_task(
                        bounded(self._run_one(s, adapter_name, adapter, t))))

        results: list[ScenarioResult] = []
        for fut in asyncio.as_completed(tasks):
            r = await fut
            if r is _SKIP:
                continue
            results.append(r)
            await _maybe_call(on_result, r)
        return results
