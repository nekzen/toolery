"""compare_summary_json(): machine-readable comparison export."""
import json
from datetime import UTC, datetime

from toolery.compare import compare_runs, compare_summary_json
from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _make_run(store: Store, run_id: str, model: str, scenarios_pass: dict[str, bool]):
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, "raw", "0.1")
    for sid, passed in scenarios_pass.items():
        result = ScenarioResult(
            scenario_id=sid, adapter="raw", trial_index=0,
            status="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None if passed else "wrong_tool",
            checks=[], trace=_trace(sid, "raw"),
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=[],
            ranking_dims=["overall"],
            scenario_hash="h", category="tool_selection", tier="easy",
            trace_path="x.json",
        )
    store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)


def test_compare_summary_json_is_valid_json_with_expected_shape(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _make_run(store, "A", "model-a", {f"easy-{i:02d}-x": True for i in range(5)} | {"easy-hard-x": False})
    _make_run(store, "B", "model-b", {f"easy-{i:02d}-x": True for i in range(5)} | {"easy-hard-x": True})

    raw = compare_summary_json(store=store, run_a="A", run_b="B")
    data = json.loads(raw)

    assert data["run_a"]["run_id"] == "A"
    assert data["run_a"]["model"] == "model-a"
    assert data["run_b"]["run_id"] == "B"
    assert data["run_b"]["model"] == "model-b"
    assert data["common_adapter"] == "raw"
    assert data["common_scenarios"] == 6
    assert len(data["metrics"]) == 1
    assert data["metrics"][0]["name"] == "Overall"
    # B passes "easy-hard-x" where A fails it -> B is better here, which this
    # module's convention records as a *regression* (B -> A got worse).
    assert len(data["regressions"]) == 1
    assert data["regressions"][0]["scenario_id"] == "easy-hard-x"
    assert data["improvements"] == []


def test_compare_summary_json_matches_markdown_report_numbers(tmp_path):
    """The JSON export and the markdown report must agree on overall scores —
    they're both built from the same compare_summary()."""
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _make_run(store, "A", "m", {f"easy-{i:02d}-x": True for i in range(3)})
    _make_run(store, "B", "m", {f"easy-{i:02d}-x": i != 0 for i in range(3)})

    md_path = tmp_path / "cmp.md"
    compare_runs(store=store, run_a="A", run_b="B", out_path=md_path)
    md = md_path.read_text()

    data = json.loads(compare_summary_json(store=store, run_a="A", run_b="B"))
    a_pct = f"{data['metrics'][0]['a']*100:.1f}"
    b_pct = f"{data['metrics'][0]['b']*100:.1f}"
    assert a_pct in md
    assert b_pct in md
