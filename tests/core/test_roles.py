from __future__ import annotations

from datetime import UTC, datetime

import pytest

from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.roles import (
    ROLES,
    RoleCheckResult,
    category_pass_rates,
    check_role,
    get_role,
    list_roles,
)
from toolery.core.store import Store


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _seed_run(store: Store, run_id: str, model: str, adapter: str,
              category_scores: dict[str, list[float]]) -> None:
    """category_scores: {category: [score, score, ...]} — each score becomes
    one scenario_results row, status='pass' when score >= 1.0."""
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


def test_roles_dict_has_expected_keys():
    for key in ("coder", "orchestrator", "fact_checker", "security_auditor",
                "creative_writer", "data_analyst", "general_assistant"):
        assert key in ROLES
        role = ROLES[key]
        assert role.name
        assert role.description
        assert role.required
        assert role.weights


def test_get_role_and_list_roles():
    assert get_role("coder") is ROLES["coder"]
    assert get_role("nonexistent_role") is None
    assert set(r.key for r in list_roles()) == set(ROLES.keys())


def test_category_pass_rates_includes_overall_bucket():
    results = [
        {"category": "coding", "status": "pass"},
        {"category": "coding", "status": "fail"},
        {"category": "debugging", "status": "pass"},
    ]
    rates = category_pass_rates(results)
    assert rates["coding"] == (0.5, 2)
    assert rates["debugging"] == (1.0, 1)
    assert rates["overall"] == (2 / 3, 3)


def test_check_role_adequate_when_thresholds_met(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_run(store, "r1", "good_model", "raw", {
        "coding": [1.0, 1.0, 1.0, 0.0],       # 75% >= 70% required
        "debugging": [1.0, 1.0, 0.0],          # 67% >= 60% required
        "code_review": [1.0, 1.0, 1.0, 0.0],   # 75% >= 65% required
    })
    result = check_role(store, "r1", "coder")
    assert isinstance(result, RoleCheckResult)
    assert result.adequate
    for c in result.checks:
        assert c.passed


def test_check_role_not_adequate_when_a_threshold_missed(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_run(store, "r2", "weak_model", "raw", {
        "coding": [1.0, 0.0, 0.0, 0.0],   # 25% < 70% required → fails
        "debugging": [1.0, 1.0, 0.0],
        "code_review": [1.0, 1.0, 1.0, 0.0],
    })
    result = check_role(store, "r2", "coder")
    assert not result.adequate
    coding_check = next(c for c in result.checks if c.category == "coding")
    assert not coding_check.passed
    assert coding_check.actual_pass_rate == pytest.approx(0.25)


def test_check_role_missing_category_counts_as_fail(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    # No security_audit results recorded at all.
    _seed_run(store, "r3", "narrow_model", "raw", {
        "code_review": [1.0, 1.0, 1.0, 1.0],
        "adversarial_robustness": [1.0, 1.0, 1.0],
    })
    result = check_role(store, "r3", "security_auditor")
    sec_check = next(c for c in result.checks if c.category == "security_audit")
    assert sec_check.actual_pass_rate is None
    assert not sec_check.passed
    assert not result.adequate


def test_check_role_unknown_role_raises(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    with pytest.raises(KeyError):
        check_role(store, "does-not-matter", "not_a_real_role")
