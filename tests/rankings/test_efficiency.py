"""`toolery efficiency`: capability next to cost, no thresholds."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from typer.testing import CliRunner

from toolery.cli import app
from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store
from toolery.rankings.efficiency import compute_efficiency


def _trace(sid: str) -> TraceResult:
    return TraceResult(scenario_id=sid, adapter="raw", trial_index=0,
                       messages=[Message(role="user", content="q")], tool_calls=[],
                       final_response="ok", started_at_iso="2026-11-04T00:00:00Z", duration_ms=1)


def _seed(store: Store, run_id: str, model: str, trials: list[tuple]) -> None:
    """trials: (category, status, correctness, calls, budget, tokens, latency_ms)."""
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(), config_json="{}",
                     scenarios_hash="h")
    for i, (cat, status, corr, calls, budget, tokens, ms) in enumerate(trials):
        sid = f"s-{i:02d}"
        store.write_scenario_result(run_id, ScenarioResult(
            scenario_id=sid, adapter="raw", trial_index=0, status=status,
            score=1.0 if status == "pass" else 0.0, correctness_score=corr,
            call_count=calls, budget_max=budget, latency_ms=ms, failure_kind=None,
            checks=[], trace=_trace(sid), completion_tokens=tokens, gen_ms=ms),
            tags=[], ranking_dims=["overall"], scenario_hash="h", category=cat,
            tier="easy", trace_path="t.json")


def _store(tmp_path) -> Store:
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    # A thinker: solves everything, but over budget and expensive.
    _seed(store, "r-think", "thinker", [
        ("coding", "fail", 1.0, 4, 2, 900, 60_000),
        ("coding", "pass", 1.0, 2, 2, 700, 40_000),
        ("debugging", "pass", 1.0, 1, 2, 500, 20_000),
        ("creative_writing", "pass", 1.0, 0, 0, 300, 10_000),
    ])
    # A sprinter: cheap and within budget, solves half.
    _seed(store, "r-fast", "sprinter", [
        ("coding", "pass", 1.0, 1, 2, 100, 2_000),
        ("coding", "fail", 0.0, 2, 2, 80, 1_000),
        ("debugging", "fail", 0.0, 1, 2, 90, 1_500),
        ("creative_writing", "pass", 1.0, 0, 0, 120, 3_000),
    ])
    return store


def test_capability_strict_overbudget_and_cost(tmp_path):
    rows = {r.model: r for r in compute_efficiency(_store(tmp_path))}
    t, s = rows["thinker"], rows["sprinter"]
    assert (t.capability, t.strict) == (1.0, 0.75)
    assert t.over_budget == 1 / 3            # zero-budget trials don't count
    assert t.tokens_per_solved == 600        # median of 900, 700, 500, 300
    assert t.seconds_per_solved == 30.0
    assert (s.capability, s.strict, s.over_budget) == (0.5, 0.5, 0.0)
    assert s.tokens_per_solved == 110        # solved trials only: 100, 120
    assert [r.model for r in compute_efficiency(_store(tmp_path / "b"))] == ["thinker", "sprinter"]


def test_category_and_run_filters(tmp_path):
    store = _store(tmp_path)
    (row,) = [r for r in compute_efficiency(store, categories=["debugging"]) if r.model == "thinker"]
    assert row.trials == 1 and row.over_budget == 0.0
    assert [r.model for r in compute_efficiency(store, run_ids=["r-fast"])] == ["sprinter"]


def test_cli_role_filter_and_json(tmp_path, monkeypatch):
    _store(tmp_path)
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    out = CliRunner().invoke(app, ["efficiency", "--role", "coder", "--json"])
    assert out.exit_code == 0, out.output
    rows = {r["model"]: r for r in json.loads(out.output)}
    assert rows["thinker"]["trials"] == 3      # coding + debugging; creative excluded
    out = CliRunner().invoke(app, ["efficiency", "--role", "nope"])
    assert out.exit_code == 2
