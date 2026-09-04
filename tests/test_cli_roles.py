from __future__ import annotations

from datetime import UTC, datetime

from typer.testing import CliRunner

from toolery.cli import app
from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store

runner = CliRunner()


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _seed_run(results_dir, run_id, model, category_scores):
    store = Store(results_dir / "runs.db")
    store.init_schema()
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, "raw", "0.1")
    i = 0
    for category, scores in category_scores.items():
        for s in scores:
            sid = f"{category}-{i:03d}-test"
            i += 1
            tr = _trace(sid, "raw")
            result = ScenarioResult(
                scenario_id=sid, adapter="raw", trial_index=0,
                status="pass" if s >= 1.0 else "fail", score=s,
                call_count=1, budget_max=1, latency_ms=10,
                failure_kind=None if s >= 1.0 else "wrong_tool",
                checks=[], trace=tr,
            )
            store.write_scenario_result(
                run_id=run_id, result=result, tags=[category],
                ranking_dims=["overall"],
                scenario_hash="h", category=category, tier="easy",
                trace_path="x.json",
            )
    store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)


def test_roles_list_shows_all_roles(monkeypatch, tmp_path):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    result = runner.invoke(app, ["roles", "list"])
    assert result.exit_code == 0
    assert "coder" in result.output
    assert "security_auditor" in result.output


def test_roles_check_adequate(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(results_dir))
    _seed_run(results_dir, "r1", "good_model", {
        "coding": [1.0, 1.0, 1.0, 0.0],
        "debugging": [1.0, 1.0, 0.0],
        "code_review": [1.0, 1.0, 1.0, 0.0],
    })
    result = runner.invoke(app, ["roles", "check", "r1", "coder"])
    assert result.exit_code == 0
    assert "ADEQUATE" in result.output


def test_roles_check_not_adequate_exits_nonzero(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(results_dir))
    _seed_run(results_dir, "r2", "weak_model", {
        "coding": [0.0, 0.0, 0.0, 0.0],
    })
    result = runner.invoke(app, ["roles", "check", "r2", "coder"])
    assert result.exit_code != 0
    assert "NOT ADEQUATE" in result.output


def test_roles_check_unknown_role(monkeypatch, tmp_path):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    result = runner.invoke(app, ["roles", "check", "some-run", "not_a_role"])
    assert result.exit_code != 0


def test_roles_check_json_output(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(results_dir))
    _seed_run(results_dir, "r3", "model_x", {
        "coding": [1.0, 1.0],
        "debugging": [1.0],
        "code_review": [1.0],
    })
    result = runner.invoke(app, ["roles", "check", "r3", "coder", "--json"])
    assert result.exit_code == 0
    assert '"adequate"' in result.output


def test_roles_rank(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(results_dir))
    _seed_run(results_dir, "r4", "model_y", {"coding": [1.0, 1.0]})
    result = runner.invoke(app, ["roles", "rank", "coder"])
    assert result.exit_code == 0
    assert "model_y" in result.output
