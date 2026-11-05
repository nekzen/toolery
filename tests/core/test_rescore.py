"""`toolery rescore`: replay stored traces through the current scorer."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from typer.testing import CliRunner

from toolery.cli import app
from toolery.core.models import (
    Budget,
    Category,
    Message,
    Scenario,
    ScenarioResult,
    Scoring,
    ScoringCheck,
    Tier,
    ToolCall,
    TraceResult,
)
from toolery.core.rescore import apply_rescore, rescore_store
from toolery.core.store import Store


def _scenario(pattern: str = "48750.25", max_calls: int = 1) -> Scenario:
    return Scenario(
        id="t-01-rescore", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="Total?", tools=["calculator"],
        budget=Budget(max_tool_calls=max_calls, max_turns=max(max_calls, 1), timeout_seconds=30),
        tool_responses={},
        scoring=Scoring(required=[ScoringCheck(check="response_contains", patterns=[pattern])]),
    )


def _seed(tmp_path, trace: TraceResult, status: str, correctness: float | None = 0.0):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="m", base_url="x",
                     started_at=datetime.now(UTC).isoformat(), config_json="{}",
                     scenarios_hash="h")
    trace_dir = tmp_path / "runs" / "r1" / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "t.json").write_text(trace.model_dump_json(), encoding="utf-8")
    store.write_scenario_result("r1", ScenarioResult(
        scenario_id=trace.scenario_id, adapter="raw", trial_index=0, status=status,
        score=1.0 if status == "pass" else 0.0, correctness_score=correctness,
        call_count=len(trace.tool_calls), budget_max=1, latency_ms=1,
        failure_kind=None if status == "pass" else "wrong_tool", checks=[], trace=trace),
        tags=[], ranking_dims=["overall"], scenario_hash="h", category="tool_selection",
        tier="easy", trace_path="traces/t.json")
    return store


def _trace(final: str, calls: int = 0, slack: float | None = None) -> TraceResult:
    meta = {"budget_slack": {"factor": slack, "cut": True}} if slack else {}
    return TraceResult(
        scenario_id="t-01-rescore", adapter="raw", trial_index=0,
        messages=[Message(role="user", content="Total?")],
        tool_calls=[ToolCall(index=i, name="calculator") for i in range(calls)],
        final_response=final, started_at_iso="2026-11-04T00:00:00Z", duration_ms=1,
        adapter_metadata=meta)


def test_rescore_flips_a_trial_the_fixed_scorer_now_passes(tmp_path):
    store = _seed(tmp_path, _trace("The total is **48,750.25**."), status="fail")
    rescored, skipped = rescore_store(store, tmp_path, {"t-01-rescore": _scenario()})
    assert not skipped
    (r,) = rescored
    assert r.flipped and r.old_status == "fail" and r.new.status == "pass"


def test_apply_writes_the_new_status_and_is_idempotent(tmp_path):
    store = _seed(tmp_path, _trace("The total is **48,750.25**."), status="fail")
    scen = {"t-01-rescore": _scenario()}
    rescored, _ = rescore_store(store, tmp_path, scen)
    assert apply_rescore(store, rescored) == 1
    (row,) = store.fetch_results_for_run("r1")
    assert row["status"] == "pass" and row["score"] == 1.0
    assert json.loads(row["checks_json"])[0]["result"] == "pass"
    rescored, _ = rescore_store(store, tmp_path, scen)
    assert apply_rescore(store, rescored) == 0


def test_slack_traces_keep_the_strict_score(tmp_path):
    """A trial recorded with --budget-slack is re-scored like the runner did:
    over budget -> strict fail, correctness from the finished run."""
    store = _seed(tmp_path, _trace("48750.25", calls=2, slack=2.0), status="fail")
    (r,), _ = rescore_store(store, tmp_path, {"t-01-rescore": _scenario(max_calls=1)})
    assert r.new.status == "fail" and r.new.failure_kind == "budget_violated"
    assert r.new.correctness_score == 1.0


def test_missing_trace_and_unknown_scenario_are_skipped_not_fatal(tmp_path):
    store = _seed(tmp_path, _trace("x"), status="fail")
    (tmp_path / "runs" / "r1" / "traces" / "t.json").unlink()
    rescored, skipped = rescore_store(store, tmp_path, {"t-01-rescore": _scenario()})
    assert rescored == [] and skipped == {"trace file missing": 1}
    rescored, skipped = rescore_store(store, tmp_path, {})
    assert skipped == {"scenario no longer exists": 1}


def test_cli_is_report_only_by_default(tmp_path, monkeypatch):
    _seed(tmp_path, _trace("The total is **48,750.25**."), status="fail")
    scen_dir = tmp_path / "scenarios"
    scen_dir.mkdir()
    (scen_dir / "t.yaml").write_text(_scenario().model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    args = ["rescore", "--scenarios-dir", str(scen_dir)]
    out = CliRunner().invoke(app, args, env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert "1 now pass" in out.output and "Report only" in out.output
    assert Store(tmp_path / "runs.db").fetch_results_for_run("r1")[0]["status"] == "fail"
    out = CliRunner().invoke(app, [*args, "--write"], env={"COLUMNS": "200"})
    assert "Updated 1 stored results" in out.output
    assert Store(tmp_path / "runs.db").fetch_results_for_run("r1")[0]["status"] == "pass"
