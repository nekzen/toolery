"""Rankings consistency dimension: score-variance-based synthetic column."""
from datetime import UTC, datetime
from pathlib import Path

from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store
from toolery.rankings.compute import compute_consistency_scores, regenerate_rankings


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _seed(store: Store, model: str, adapter: str, scores: list[float], run_id: str):
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, adapter, "0.1")
    for i, s in enumerate(scores):
        sid = f"easy-{i:02d}-test"
        tr = _trace(sid, adapter)
        result = ScenarioResult(
            scenario_id=sid, adapter=adapter, trial_index=0,
            status="pass" if s >= 1.0 else "fail", score=s,
            call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None if s >= 1.0 else "wrong_tool",
            checks=[], trace=tr,
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=[], ranking_dims=["overall"],
            scenario_hash="h", category="coding", tier="easy",
            trace_path="x.json",
        )
    store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)


def test_zero_variance_gives_consistency_one(tmp_path: Path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed(store, "model_steady", "raw", [1.0, 1.0, 1.0, 1.0], "r1")
    scores = compute_consistency_scores(store)
    assert scores[("model_steady", "raw")]["score"] == 1.0


def test_max_variance_gives_consistency_zero(tmp_path: Path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    # 50/50 split at the extremes 0.0/1.0 -> population stddev == 0.5 (max).
    _seed(store, "model_erratic", "raw", [0.0, 1.0, 0.0, 1.0], "r1")
    scores = compute_consistency_scores(store)
    assert abs(scores[("model_erratic", "raw")]["score"] - 0.0) < 1e-9


def test_partial_variance_gives_intermediate_score(tmp_path: Path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed(store, "model_mid", "raw", [1.0, 0.5, 1.0, 0.5], "r1")
    scores = compute_consistency_scores(store)
    v = scores[("model_mid", "raw")]
    assert 0.0 < v["score"] < 1.0


def test_regenerate_rankings_writes_consistency_md(tmp_path: Path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed(store, "model_a", "raw", [1.0, 1.0, 1.0], "r1")
    _seed(store, "model_b", "raw", [0.0, 1.0, 0.0], "r2")
    out_dir = tmp_path / "rankings"
    regenerate_rankings(store=store, dimensions=["consistency"], out_dir=out_dir)
    md = (out_dir / "consistency.md").read_text()
    assert "model_a" in md and "model_b" in md
    # model_a (zero variance) must rank ahead of model_b (high variance).
    assert md.index("model_a") < md.index("model_b")


def test_regenerate_rankings_consistency_alongside_overall(tmp_path: Path):
    """'consistency' must not interfere with the normal per-dimension loop
    when both are requested together."""
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed(store, "model_a", "raw", [1.0, 1.0], "r1")
    out_dir = tmp_path / "rankings"
    regenerate_rankings(store=store, dimensions=["overall", "consistency"], out_dir=out_dir)
    assert (out_dir / "overall.md").exists()
    assert (out_dir / "consistency.md").exists()
