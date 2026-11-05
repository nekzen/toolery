"""Capability vs efficiency, side by side, per model.

A budget is a line somebody drew; whether 3 tool calls is "enough" depends
on the job. This view draws no line: it puts what a model can do next to
what it costs to get there, so models can be weighed against each other.

- capability: share of trials whose task was solved, budget ignored
  (correctness_score).
- strict:     share of trials passed within the budget.
- over_budget: share of trials (with a budget > 0) that used more tool calls
  than the budget.
- tokens/seconds per solved trial: medians over solved trials. Tokens compare
  across machines; seconds only on the same hardware and serving setup.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from toolery.core.store import Store


@dataclass(frozen=True)
class EfficiencyRow:
    model: str
    adapter: str
    runs: int
    trials: int
    capability: float
    strict: float
    over_budget: float | None
    tokens_per_solved: float | None
    seconds_per_solved: float | None


def compute_efficiency(store: Store, *, run_ids: list[str] | None = None,
                       categories: list[str] | None = None) -> list[EfficiencyRow]:
    """One row per (model, adapter) over the selected runs (default: every
    run that is not soft-deleted) and categories (default: all)."""
    runs = {r["run_id"]: r for r in store.fetch_all_runs()}
    wanted = set(run_ids) if run_ids else set(runs)
    cats = set(categories) if categories else None
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    run_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for run_id in wanted:
        meta = runs.get(run_id)
        if meta is None:
            continue
        for r in store.fetch_results_for_run(run_id):
            if cats is not None and r.get("category") not in cats:
                continue
            key = (meta["model"], r["adapter"])
            groups[key].append(r)
            run_sets[key].add(run_id)

    out: list[EfficiencyRow] = []
    for (model, adapter), rs in groups.items():
        solved = [r for r in rs if (r.get("correctness_score") or 0) >= 1.0]
        budgeted = [r for r in rs if (r.get("budget_max") or 0) > 0]
        tokens = [r["completion_tokens"] for r in solved if r.get("completion_tokens")]
        seconds = [r["latency_ms"] / 1000 for r in solved if r.get("latency_ms")]
        out.append(EfficiencyRow(
            model=model, adapter=adapter, runs=len(run_sets[(model, adapter)]),
            trials=len(rs),
            capability=len(solved) / len(rs),
            strict=sum((r.get("status") or "") == "pass" for r in rs) / len(rs),
            over_budget=(sum(r["call_count"] > r["budget_max"] for r in budgeted) / len(budgeted)
                         if budgeted else None),
            tokens_per_solved=statistics.median(tokens) if tokens else None,
            seconds_per_solved=statistics.median(seconds) if seconds else None,
        ))
    out.sort(key=lambda row: (-row.capability, row.model))
    return out
