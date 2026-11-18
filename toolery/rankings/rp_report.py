"""Long-context writing / roleplay results, per probe and context length.

The rp_long_context scenarios are sampled at a writing temperature: a defect
shows up in some trials and not in others, so the result is a RATE — "leaked
the name in 2 of 10 trials at 16k" — not a pass or a fail. This module reads
those rates out of runs.db; nothing here is a ranking.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from toolery.core.store import Store

CATEGORY = "rp_long_context"
_LENGTH_ORDER = {"8k": 0, "16k": 1, "32k": 2, "64k": 3}


@dataclass
class RpCell:
    """One model × adapter × probe × context length."""
    model: str
    adapter: str
    probe: str
    length: str
    trials: int = 0
    passed: int = 0
    errors: int = 0                       # no reply at all (error / timeout)
    failures: dict[str, int] = field(default_factory=dict)   # failure_kind -> count
    prompt_tokens: float | None = None    # mean, as counted by the server
    sampling: dict = field(default_factory=dict)
    conversation_sha256: str | None = None
    runs: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)       # failed trials
    # Medians of the figures a measuring check reports (repetition_bounded).
    metrics: dict = field(default_factory=dict)

    @property
    def judged(self) -> int:
        return self.trials - self.errors

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.judged if self.judged else None


def length_key(length: str) -> tuple[int, str]:
    return (_LENGTH_ORDER.get(length, 99), length)


def _proof(checks_json: str | None) -> str:
    """The spoken line quoted by the check that failed the trial: a forbidden
    check that fired is stored inverted, as a failure."""
    try:
        checks = json.loads(checks_json or "[]")
    except ValueError:
        return ""
    for c in checks if isinstance(checks, list) else []:
        detail = str(c.get("detail", "")) if isinstance(c, dict) else ""
        if c.get("result") == "fail" and detail.startswith("(inverted) spoken:"):
            return detail.removeprefix("(inverted) ")
    return ""


def _metrics(checks_json: str | None) -> dict | None:
    """Figures reported by a measuring check: JSON after "repetition: "."""
    try:
        checks = json.loads(checks_json or "[]")
    except ValueError:
        return None
    for c in checks if isinstance(checks, list) else []:
        detail = str(c.get("detail", "")) if isinstance(c, dict) else ""
        _, sep, payload = detail.partition("repetition: ")
        if sep:
            try:
                return json.loads(payload)
            except ValueError:
                return None
    return None


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 4)


def compute_rp_report(store: Store, *, run_ids: list[str] | None = None,
                      models: list[str] | None = None) -> list[RpCell]:
    runs = {r["run_id"]: r for r in store.fetch_all_runs()}
    wanted = set(run_ids) if run_ids else set(runs)
    cells: dict[tuple[str, str, str, str], RpCell] = {}
    tokens: dict[tuple, list[int]] = defaultdict(list)
    kinds: dict[tuple, Counter] = defaultdict(Counter)
    measured: dict[tuple, list[dict]] = defaultdict(list)
    for run_id in sorted(wanted):
        run = runs.get(run_id)
        if run is None or (models and run["model"] not in models):
            continue
        for r in store.fetch_results_for_run(run_id):
            if r.get("category") != CATEGORY:
                continue
            meta = json.loads(r.get("meta_json") or "{}")
            conv = meta.get("conversation") or {}
            length = conv.get("length") or "?"
            probe = r["scenario_id"].removeprefix("rp-").removesuffix(f"-{length}")
            key = (run["model"], r["adapter"], probe, length)
            cell = cells.setdefault(key, RpCell(*key))
            cell.trials += 1
            if run_id not in cell.runs:
                cell.runs.append(run_id)
            cell.sampling = {k: v for k, v in (meta.get("sampling") or {}).items() if k != "seed"}
            cell.conversation_sha256 = conv.get("sha256")
            if r.get("prompt_tokens"):
                tokens[key].append(r["prompt_tokens"])
            m = _metrics(r.get("checks_json"))
            if m:
                measured[key].append(m)
            if r["status"] in ("error", "timeout"):
                cell.errors += 1
                kinds[key][r.get("failure_kind") or r["status"]] += 1
            elif r["status"] == "pass":
                cell.passed += 1
            else:
                kind = r.get("failure_kind") or "failed"
                kinds[key][kind] += 1
                cell.examples.append({
                    "run_id": run_id, "scenario_id": r["scenario_id"],
                    "trial": r["trial_index"], "failure_kind": kind,
                    "seed": (meta.get("sampling") or {}).get("seed"),
                    "trace_path": r.get("trace_path"), "proof": _proof(r.get("checks_json")),
                })
    for key, cell in cells.items():
        cell.failures = dict(kinds[key].most_common())
        if tokens[key]:
            cell.prompt_tokens = sum(tokens[key]) / len(tokens[key])
        ms = measured[key]
        if ms:
            tic_names = sorted({name for m in ms for name in m.get("tics_per_1000", {})})
            cell.metrics = {
                "trials": len(ms),
                "ngram_share": _median([m["ngram_share"] for m in ms]),
                "same_opener": _median([m["same_opener"] for m in ms]),
                "words": _median([m["words"] for m in ms]),
                # Totals, not medians: cut replies over all replies of the cell.
                # None when no trial reported it (runs scored before the count).
                "truncated": (sum(m["truncated"] for m in ms if m.get("truncated") is not None)
                              if any(m.get("truncated") is not None for m in ms) else None),
                "replies": sum(m.get("replies", 0) for m in ms),
                "tics_per_1000": {n: _median([m.get("tics_per_1000", {}).get(n, 0.0) for m in ms])
                                  for n in tic_names},
            }
    return sorted(cells.values(),
                  key=lambda c: (c.model, c.adapter, c.probe, length_key(c.length)))
