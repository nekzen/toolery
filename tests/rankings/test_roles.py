from __future__ import annotations

from datetime import UTC, datetime

from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store
from toolery.rankings.roles import (
    RoleRankingRow,
    compute_role_ranking,
    render_role_check_md,
    render_role_ranking_md,
)


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _seed_run(store: Store, run_id: str, model: str, adapter: str,
              category_scores: dict[str, list[float]]) -> None:
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, adapter, "0.1")
    i = 0
    for category, scores in category_scores.items():
        for s in scores:
            sid = f"{category}-{i:03d}-test"
            i += 1
            tr = _trace(sid, adapter)
            result = ScenarioResult(
                scenario_id=sid, adapter=adapter, trial_index=0,
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


def test_compute_role_ranking_orders_by_weighted_score(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_run(store, "r1", "strong_coder", "raw", {
        "coding": [1.0, 1.0, 1.0, 1.0],
        "debugging": [1.0, 1.0],
    })
    _seed_run(store, "r2", "weak_coder", "raw", {
        "coding": [1.0, 0.0, 0.0, 0.0],
        "debugging": [0.0, 0.0],
    })
    rows = compute_role_ranking(store, "coder")
    assert len(rows) == 2
    assert isinstance(rows[0], RoleRankingRow)
    assert rows[0].model == "strong_coder"
    assert rows[0].weighted_score > rows[1].weighted_score


def test_compute_role_ranking_unknown_role_raises(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    import pytest
    with pytest.raises(KeyError):
        compute_role_ranking(store, "not_a_role")


def test_render_role_ranking_md_contains_models_and_thresholds(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_run(store, "r1", "model_a", "raw", {"coding": [1.0, 1.0]})
    rows = compute_role_ranking(store, "coder")
    from toolery.core.roles import get_role
    role = get_role("coder")
    md = render_role_ranking_md(role, rows)
    assert "Coder Ranking" in md
    assert "model_a" in md
    assert "coding" in md  # required-thresholds table


def test_render_role_check_md_shows_verdict(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_run(store, "r1", "model_a", "raw", {
        "coding": [1.0, 1.0, 1.0, 1.0],
        "debugging": [1.0, 1.0],
        "code_review": [1.0, 1.0],
    })
    from toolery.core.roles import check_role
    result = check_role(store, "r1", "coder")
    md = render_role_check_md(result)
    assert "Role check" in md
    assert "Verdict:" in md
    assert ("ADEQUATE" in md)
