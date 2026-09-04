"""Runner retry logic for transient adapter failures.

Retry must fire on transient errors (429/timeout/connection reset/5xx) and
must NEVER retry genuine model failures (bad tool call, wrong answer, or any
other non-transient error string).
"""
import pytest

from toolery.core.models import Budget, Category, Scenario, Scoring, ScoringCheck, Tier, TraceResult
from toolery.core.runner import Runner, is_transient_error


def _scenario():
    return Scenario(
        id="easy-01-direct-weather", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="p", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring=Scoring(
            required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
        ),
    )


class _FlakyAdapter:
    """Fails with a transient error `fail_times` times, then succeeds."""
    name = "flaky"
    version = "0.1"

    def __init__(self, fail_times: int, error: str = "429 Too Many Requests"):
        self.fail_times = fail_times
        self.error = error
        self.calls = 0

    async def run_scenario(self, scenario, model, timeout):
        self.calls += 1
        if self.calls <= self.fail_times:
            return TraceResult(
                scenario_id=scenario.id, adapter=self.name, trial_index=0,
                messages=[], tool_calls=[], final_response=None,
                started_at_iso="2026-01-01T00:00:00Z", duration_ms=1,
                error=self.error,
            )
        from toolery.core.models import ToolCall
        return TraceResult(
            scenario_id=scenario.id, adapter=self.name, trial_index=0,
            messages=[], final_response="ok",
            tool_calls=[ToolCall(index=0, name="get_weather", args={})],
            started_at_iso="2026-01-01T00:00:00Z", duration_ms=1, error=None,
        )


# ── is_transient_error classification ───────────────────────────────────────

@pytest.mark.parametrize("error", [
    "429 Too Many Requests", "rate limit exceeded", "timeout",
    "Connection reset by peer", "connection refused", "502 Bad Gateway",
    "503 Service Unavailable", "504 Gateway Timeout", "HTTPError: 500 Internal Server Error",
])
def test_is_transient_error_matches_transient(error):
    assert is_transient_error(error)


@pytest.mark.parametrize("error", [
    "unknown tool called: foo", "schema violation: missing field 'x'",
    "wrong answer", "KeyError: 'choices'", "invalid json", "model_crash",
])
def test_is_transient_error_never_matches_genuine_failures(error):
    assert not is_transient_error(error)


# ── Runner integration ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_retries_transient_error_and_eventually_succeeds():
    adapter = _FlakyAdapter(fail_times=2, error="429 Too Many Requests")
    runner = Runner(adapters={"flaky": adapter}, trials=1, model="x",
                    max_retries=3, retry_backoff_base=0.001, retry_backoff_max=0.01)
    results = await runner.run([_scenario()])
    assert len(results) == 1
    assert results[0].status == "pass"
    assert adapter.calls == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_runner_gives_up_after_max_retries_exhausted():
    adapter = _FlakyAdapter(fail_times=10, error="503 Service Unavailable")
    runner = Runner(adapters={"flaky": adapter}, trials=1, model="x",
                    max_retries=2, retry_backoff_base=0.001, retry_backoff_max=0.01)
    results = await runner.run([_scenario()])
    assert len(results) == 1
    assert results[0].status == "error"
    assert adapter.calls == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_runner_never_retries_genuine_model_failure():
    adapter = _FlakyAdapter(fail_times=10, error="unknown tool called: bogus_tool")
    runner = Runner(adapters={"flaky": adapter}, trials=1, model="x",
                    max_retries=5, retry_backoff_base=0.001, retry_backoff_max=0.01)
    results = await runner.run([_scenario()])
    assert len(results) == 1
    assert results[0].status == "error"
    assert adapter.calls == 1  # no retry attempted


@pytest.mark.asyncio
async def test_runner_default_max_retries_is_zero_no_behavior_change():
    adapter = _FlakyAdapter(fail_times=1, error="429 Too Many Requests")
    runner = Runner(adapters={"flaky": adapter}, trials=1, model="x")
    results = await runner.run([_scenario()])
    assert results[0].status == "error"
    assert adapter.calls == 1
