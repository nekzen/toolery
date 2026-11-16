"""Long-context writing / roleplay results, per probe and context length.

The rp_long_context scenarios are sampled at a writing temperature: a defect
shows up in some trials and not in others, so the result is a RATE — "leaked
the name in 2 of 10 trials at 16k" — not a pass or a fail. This module reads
those rates out of runs.db; nothing here is a ranking.
"""

from __future__ import annotations

import json
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

    @property
    def judged(self) -> int:
        return self.trials - self.errors

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.judged if self.judged else None


def length_key(length: str) -> tuple[int, str]:
    return (_LENGTH_ORDER.get(length, 99), length)


def _proof(checks_json: str | None) -> str:
    """The spoken line a dialogue check quoted, when there is one."""
    try:
        checks = json.loads(checks_json or "[]")
    except ValueError:
        return ""
    for c in checks if isinstance(checks, list) else []:
        detail = str(c.get("detail", "")) if isinstance(c, dict) else ""
        if detail.startswith("spoken:") and len(detail) > 12:
            return detail
    return ""


def compute_rp_report(store: Store, *, run_ids: list[str] | None = None,
                      models: list[str] | None = None) -> list[RpCell]:
    runs = {r["run_id"]: r for r in store.fetch_all_runs()}
    wanted = set(run_ids) if run_ids else set(runs)
    cells: dict[tuple[str, str, str, str], RpCell] = {}
    tokens: dict[tuple, list[int]] = defaultdict(list)
    kinds: dict[tuple, Counter] = defaultdict(Counter)
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
    return sorted(cells.values(),
                  key=lambda c: (c.model, c.adapter, c.probe, length_key(c.length)))
