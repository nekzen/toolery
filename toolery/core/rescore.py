"""Re-score stored trials with the current scorer and scenario files.

No inference: each trial's saved trace is replayed through today's
evaluate(), so a scorer or check fix can be measured — and applied — on
every past run. Trials recorded with --budget-slack are re-scored the same
way the runner scores them (strict score on strict_view, correctness on the
full trace).

Exactness: a change to scoring (checks, patterns, the scorer) is replayed
exactly. A change to what the model *saw* (mock responses, the prompt, the
tool list, a larger budget than the trace was recorded with) is not: the
model might have acted differently, so those re-scores are estimates and
such scenarios deserve a fresh run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from toolery.core.models import Scenario, ScenarioResult, TraceResult
from toolery.core.runner import _evaluate_with_slack
from toolery.core.scorer import evaluate
from toolery.core.store import Store


@dataclass(frozen=True)
class Rescored:
    result_id: int
    run_id: str
    model: str
    scenario_id: str
    adapter: str
    trial_index: int
    old_status: str | None
    old_score: float
    old_correctness: float | None
    new: ScenarioResult

    @property
    def flipped(self) -> bool:
        return (self.old_status or "") != self.new.status

    @property
    def correctness_changed(self) -> bool:
        return self.old_correctness != self.new.correctness_score


def rescore_trace(scenario: Scenario, trace: TraceResult) -> ScenarioResult:
    """Score a stored trace the way the runner scored it when it was recorded."""
    slack = (trace.adapter_metadata or {}).get("budget_slack")
    if isinstance(slack, dict) and float(slack.get("factor", 1.0)) > 1.0:
        return _evaluate_with_slack(scenario, trace, float(slack["factor"]))
    return evaluate(scenario, trace)


def rescore_store(store: Store, results_dir: Path, scenarios: dict[str, Scenario],
                  run_ids: list[str] | None = None,
                  scenario_ids: list[str] | None = None) -> tuple[list[Rescored], dict[str, int]]:
    """Re-score every stored trial (optionally filtered). Returns the rescored
    trials and a count of skipped ones by reason."""
    skipped: dict[str, int] = defaultdict(int)
    out: list[Rescored] = []
    runs = {r["run_id"]: r for r in store.fetch_all_runs()}
    wanted_runs = set(run_ids) if run_ids else set(runs)
    wanted_sc = set(scenario_ids) if scenario_ids else None
    for run_id in sorted(wanted_runs):
        meta = runs.get(run_id)
        if meta is None:
            skipped["unknown run"] += 1
            continue
        for row in store.fetch_results_for_run(run_id):
            sid = row["scenario_id"]
            if wanted_sc is not None and sid not in wanted_sc:
                continue
            scenario = scenarios.get(sid)
            if scenario is None:
                skipped["scenario no longer exists"] += 1
                continue
            tp = results_dir / "runs" / run_id / (row.get("trace_path") or "")
            if not row.get("trace_path") or not tp.is_file():
                skipped["trace file missing"] += 1
                continue
            try:
                trace = TraceResult.model_validate_json(tp.read_text(encoding="utf-8"))
            except ValueError:
                skipped["trace file unreadable"] += 1
                continue
            new = rescore_trace(scenario, trace)
            out.append(Rescored(
                result_id=row["result_id"], run_id=run_id, model=meta["model"],
                scenario_id=sid, adapter=row["adapter"], trial_index=row["trial_index"],
                old_status=row["status"], old_score=row["score"],
                old_correctness=row.get("correctness_score"), new=new,
            ))
    return out, dict(skipped)


def apply_rescore(store: Store, rescored: list[Rescored]) -> int:
    """Write the new scores back to scenario_results. Returns rows changed."""
    changed = 0
    with store.conn() as c:
        for r in rescored:
            n = r.new
            if not r.flipped and r.old_score == n.score and not r.correctness_changed:
                continue
            c.execute(
                "UPDATE scenario_results SET status=?, score=?, failure_kind=?, "
                "correctness_score=?, checks_json=? WHERE result_id=?",
                (n.status, n.score, n.failure_kind, n.correctness_score,
                 json.dumps([ck.model_dump() for ck in n.checks]), r.result_id),
            )
            changed += 1
    return changed
