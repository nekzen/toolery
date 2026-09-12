"""`toolery correctness-report` must actually print its table (it used to
build it and return without printing anything)."""
from __future__ import annotations

from datetime import UTC, datetime

from typer.testing import CliRunner

from toolery.cli import app
from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store


def _seed(results_dir):
    store = Store(results_dir / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="demo-model", base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    trace = TraceResult(scenario_id="s1", adapter="raw", trial_index=0,
                        messages=[Message(role="user", content="hi")], tool_calls=[],
                        final_response="ok", started_at_iso="2026-09-12T00:00:00Z",
                        duration_ms=1, error=None)
    # Over budget (score 0) but correct once the budget is ignored.
    r = ScenarioResult(scenario_id="s1", adapter="raw", trial_index=0, status="fail",
                       score=0.0, correctness_score=1.0, call_count=3, budget_max=2,
                       latency_ms=1, failure_kind="budget_violated", checks=[], trace=trace)
    store.write_scenario_result("r1", r, tags=[], ranking_dims=["overall"],
                                scenario_hash="h", category="coding", tier="easy",
                                trace_path="t.json")


def test_correctness_report_prints_the_table(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    _seed(tmp_path)
    out = CliRunner().invoke(app, ["correctness-report"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert "demo-model" in out.output
    assert "1.000" in out.output          # correctness
    assert "0.000" in out.output          # budgeted score


def test_correctness_report_says_so_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    out = CliRunner().invoke(app, ["correctness-report"])
    assert out.exit_code == 0
    assert "No correctness data" in out.output
