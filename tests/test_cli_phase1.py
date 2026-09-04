"""CLI --json / --dry-run / delete-run / restore-run coverage (Phase 1)."""
import json

from typer.testing import CliRunner

from toolery.cli import app

runner = CliRunner()


def test_cli_list_json_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == []


def test_cli_list_json_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    from toolery.core.store import Store
    store = Store(tmp_path / "results" / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="m", base_url="u",
                     started_at="2026-01-01T00:00:00Z", config_json="{}", scenarios_hash="h")

    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["run_id"] == "r1"
    assert data[0]["model"] == "m"


def test_cli_list_json_excludes_deleted_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    from toolery.core.store import Store
    store = Store(tmp_path / "results" / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="m", base_url="u",
                     started_at="2026-01-01T00:00:00Z", config_json="{}", scenarios_hash="h")
    store.delete_run("r1")

    result = runner.invoke(app, ["list", "--json"])
    assert json.loads(result.output) == []

    result_all = runner.invoke(app, ["list", "--json", "--include-deleted"])
    data = json.loads(result_all.output)
    assert len(data) == 1
    assert data[0]["run_id"] == "r1"


def test_cli_delete_run_and_restore_run_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    from toolery.core.store import Store
    store = Store(tmp_path / "results" / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="m", base_url="u",
                     started_at="2026-01-01T00:00:00Z", config_json="{}", scenarios_hash="h")

    result = runner.invoke(app, ["delete-run", "r1"])
    assert result.exit_code == 0, result.output
    assert Store(tmp_path / "results" / "runs.db").fetch_all_runs() == []

    result = runner.invoke(app, ["restore-run", "r1"])
    assert result.exit_code == 0, result.output
    active = Store(tmp_path / "results" / "runs.db").fetch_all_runs()
    assert [r["run_id"] for r in active] == ["r1"]


def test_cli_delete_run_missing_run_id_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    result = runner.invoke(app, ["delete-run", "nope"])
    assert result.exit_code != 0


def test_cli_scenarios_json(tmp_path, monkeypatch):
    result = runner.invoke(app, ["scenarios", "--tier", "easy", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert all("id" in row and "tier" in row and "tools" in row for row in data)
    assert all(row["tier"] == "easy" for row in data)


def test_cli_run_dry_run_reports_plan_without_executing(tmp_path, monkeypatch):
    import types

    import toolery.cli as cli_module

    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    fake_scenarios = [
        types.SimpleNamespace(
            id=f"easy-{i:02d}-x", title="t",
            tier=types.SimpleNamespace(value="easy"),
            category=types.SimpleNamespace(value="general"),
            budget=types.SimpleNamespace(timeout_seconds=30),
            tags=[], ranking_dimensions=[],
        )
        for i in range(3)
    ]
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: fake_scenarios)

    called = {"runner_constructed": False}

    class _StubRunner:
        def __init__(self, *a, **kw):
            called["runner_constructed"] = True

        async def run(self, *a, **kw):
            return []

    monkeypatch.setattr(cli_module, "Runner", _StubRunner)

    result = runner.invoke(app, [
        "run", "--model", "any-model", "--adapter", "raw", "--tier", "easy",
        "--trials", "2", "--concurrency", "1", "--base-url", "http://localhost:0",
        "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not called["runner_constructed"]
    # No run should have been persisted to the DB — dry-run never opens the store.
    db_path = tmp_path / "results" / "runs.db"
    if db_path.exists():
        from toolery.core.store import Store
        store = Store(db_path)
        store.init_schema()
        assert store.fetch_all_runs() == []


def test_cli_run_dry_run_json_reports_total_units(tmp_path, monkeypatch):
    import types

    import toolery.cli as cli_module

    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    fake_scenarios = [
        types.SimpleNamespace(
            id=f"easy-{i:02d}-x", title="t",
            tier=types.SimpleNamespace(value="easy"),
            category=types.SimpleNamespace(value="general"),
            budget=types.SimpleNamespace(timeout_seconds=30),
            tags=[], ranking_dimensions=[],
        )
        for i in range(3)
    ]
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: fake_scenarios)

    result = runner.invoke(app, [
        "run", "--model", "any-model", "--adapter", "raw", "--tier", "easy",
        "--trials", "2", "--concurrency", "1", "--base-url", "http://localhost:0",
        "--dry-run", "--json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["scenarios_count"] == 3
    assert data["trials"] == 2
    assert data["total_units"] == 3 * 1 * 2  # scenarios * adapters * trials
    assert data["adapters"] == ["raw"]


def test_cli_run_dry_run_rejects_bad_filter(tmp_path, monkeypatch):
    import types

    import toolery.cli as cli_module

    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: [
        types.SimpleNamespace(
            id="easy-00-x", title="t",
            tier=types.SimpleNamespace(value="easy"),
            category=types.SimpleNamespace(value="general"),
            budget=types.SimpleNamespace(timeout_seconds=30),
            tags=[], ranking_dimensions=[],
        )
    ])

    result = runner.invoke(app, [
        "run", "--model", "any-model", "--adapter", "raw", "--tier", "very_hard",
        "--trials", "1", "--concurrency", "1", "--base-url", "http://localhost:0",
        "--dry-run",
    ])
    # No scenarios match the very_hard filter -> validation error before dry-run report.
    assert result.exit_code == 2


def test_cli_compare_json(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from toolery.core.models import Message, ScenarioResult, TraceResult
    from toolery.core.store import Store

    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    store = Store(tmp_path / "results" / "runs.db")
    store.init_schema()

    def _make_run(run_id, model):
        store.create_run(run_id=run_id, model=model, base_url="x",
                         started_at=datetime.now(UTC).isoformat(),
                         config_json="{}", scenarios_hash="h")
        store.upsert_adapter(run_id, "raw", "0.1")
        trace = TraceResult(
            scenario_id="easy-00-x", adapter="raw", trial_index=0,
            messages=[Message(role="user", content="hi")],
            tool_calls=[], final_response="ok",
            started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
        )
        result = ScenarioResult(
            scenario_id="easy-00-x", adapter="raw", trial_index=0,
            status="pass", score=1.0, call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None, checks=[], trace=trace,
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=[], ranking_dims=["overall"],
            scenario_hash="h", category="tool_selection", tier="easy",
            trace_path="x.json",
        )
        store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)

    _make_run("A", "model-a")
    _make_run("B", "model-b")

    result = runner.invoke(app, ["compare", "A", "B", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["run_a"]["run_id"] == "A"
    assert data["run_b"]["run_id"] == "B"
