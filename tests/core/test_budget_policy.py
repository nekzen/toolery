"""Per-category budget policy and the failure taxonomy that goes with it."""
from __future__ import annotations

import httpx
import pytest
import respx

import toolery.tools.generic  # noqa: F401 — registers get_weather
from toolery.adapters.openai_raw import OpenAIRawAdapter
from toolery.core.budget_policy import LENIENT_CAP, budget_policy
from toolery.core.models import (
    Budget,
    Category,
    Message,
    Scenario,
    Scoring,
    ScoringCheck,
    Tier,
    ToolCall,
    ToolResponseRule,
    TraceResult,
)
from toolery.core.rescore import rescore_trace
from toolery.core.runner import Runner
from toolery.core.scorer import evaluate

URL = "http://localhost:8000/v1/chat/completions"


def _scenario(category: Category = Category.DEBUGGING, calls: int = 1, *,
              dims: list[str] | None = None, enforce: str | None = None,
              forbidden: list[ScoringCheck] | None = None) -> Scenario:
    return Scenario(
        id="t-01-policy", title="t", tier=Tier.EASY, category=category,
        domain="generic", description="d", prompt="Weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=calls, max_turns=max(calls, 1), timeout_seconds=30,
                      enforce=enforce),
        ranking_dimensions=dims or ["overall"],
        tool_responses={"get_weather": [ToolResponseRule(match="any", returns={"temp_c": 7})]},
        scoring=Scoring(required=[ScoringCheck(check="response_contains", patterns=["7"])],
                        forbidden=forbidden or []),
    )


# --- resolution -----------------------------------------------------------

def test_category_defaults():
    assert budget_policy(_scenario(Category.DEBUGGING)) == "lenient"
    assert budget_policy(_scenario(Category.SAFETY_BOUNDARIES)) == "lenient"
    assert budget_policy(_scenario(Category.CODING)) == "strict"
    assert budget_policy(_scenario(Category.RESTRAINT_REFUSAL)) == "strict"


def test_absolute_rules_beat_category_and_override():
    # Zero budget: calling a tool is the failure itself.
    assert budget_policy(_scenario(Category.DEBUGGING, calls=0, enforce="lenient")) == "strict"
    # The budget is what the scenario measures.
    assert budget_policy(_scenario(Category.DEBUGGING, dims=["overall", "budget_efficiency"],
                                   enforce="lenient")) == "strict"


def test_scenario_override_beats_category_default():
    assert budget_policy(_scenario(Category.CODING, enforce="lenient")) == "lenient"
    assert budget_policy(_scenario(Category.DEBUGGING, enforce="strict")) == "strict"


# --- the runner, against a scripted server --------------------------------

def _tool_turn(n: int) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": f"c{n}", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"location": "Warsaw"}'}}]}}]})


def _answer() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": "It is 7°C."}}]})


async def _run(scenario: Scenario, n_calls: int):
    respx.post(URL).mock(side_effect=[_tool_turn(i) for i in range(n_calls)] + [_answer()])
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    (result,) = await Runner(adapters={"raw": adapter}, trials=1).run([scenario])
    return result


@pytest.mark.asyncio
@respx.mock
async def test_lenient_overrun_passes_and_is_recorded():
    r = await _run(_scenario(Category.DEBUGGING, calls=1), n_calls=2)   # 2 calls, budget 1
    assert r.status == "pass" and r.failure_kind is None
    sig = r.trace.adapter_metadata["budget_policy"]
    assert sig == {"policy": "lenient", "calls": 2, "budget_calls": 1,
                   "cap_calls": int(LENIENT_CAP), "over_budget": True}
    note = [c for c in r.checks if c.check == "budget_respected"]
    assert note and note[0].result == "fail" and "does not fail the trial" in note[0].detail


@pytest.mark.asyncio
@respx.mock
async def test_lenient_past_the_cap_still_fails():
    r = await _run(_scenario(Category.DEBUGGING, calls=1), n_calls=4)   # cap is 3
    assert r.status == "fail" and r.failure_kind == "budget_violated"


@pytest.mark.asyncio
@respx.mock
async def test_strict_category_is_unchanged():
    r = await _run(_scenario(Category.CODING, calls=1), n_calls=2)
    assert r.status == "fail" and r.failure_kind == "budget_violated"


@pytest.mark.asyncio
@respx.mock
async def test_lenient_within_budget_records_no_overrun():
    r = await _run(_scenario(Category.DEBUGGING, calls=1), n_calls=1)
    assert r.status == "pass"
    assert r.trace.adapter_metadata["budget_policy"]["over_budget"] is False
    assert not [c for c in r.checks if c.check == "budget_respected"]


# --- taxonomy -------------------------------------------------------------

def _trace(n_calls: int, final: str | None = "7", error: str | None = None) -> TraceResult:
    return TraceResult(
        scenario_id="t-01-policy", adapter="raw", trial_index=0,
        messages=[Message(role="user", content="q")],
        tool_calls=[ToolCall(index=i, name="get_weather") for i in range(n_calls)],
        final_response=final, started_at_iso="2026-11-04T00:00:00Z", duration_ms=1, error=error)


FORBID = [ScoringCheck(check="tool_called", tool="get_weather")]


def test_forbidden_action_outranks_budget_violated():
    s = _scenario(Category.CODING, calls=1, forbidden=FORBID)
    assert evaluate(s, _trace(3)).failure_kind == "forbidden_action"


def test_cut_before_any_forbidden_action_stays_budget_violated():
    s = _scenario(Category.CODING, calls=1,
                  forbidden=[ScoringCheck(check="tool_called", tool="delete_user")])
    assert evaluate(s, _trace(3)).failure_kind == "budget_violated"


def test_forbidden_action_before_a_timeout_is_reported():
    s = _scenario(Category.CODING, calls=5, forbidden=FORBID)
    r = evaluate(s, _trace(1, final=None, error="timeout: scenario time budget of 60s exhausted"))
    assert r.status == "error" and r.failure_kind == "forbidden_action"
    r = evaluate(_scenario(Category.CODING, calls=5),
                 _trace(1, final=None, error="timeout: scenario time budget of 60s exhausted"))
    assert r.failure_kind == "timeout"


# --- rescore follows the same policy --------------------------------------

def test_rescore_applies_the_policy_to_stored_traces():
    over = _trace(2).model_copy(update={"adapter_metadata": {"budget_slack": {"factor": 2.0}}})
    assert rescore_trace(_scenario(Category.DEBUGGING, calls=1), over).status == "pass"
    assert rescore_trace(_scenario(Category.CODING, calls=1), over).status == "fail"
