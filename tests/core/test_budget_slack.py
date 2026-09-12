"""Budget slack: the model may run past its budget, the strict score must be
exactly what a normal run gives, and correctness_score must see the finished
run ("it failed, but here is what it would have done")."""
from __future__ import annotations

import httpx
import pytest
import respx

import toolery.tools.generic  # noqa: F401 — registers get_weather
from toolery.adapters.base import MockAdapter, ScenarioPlan
from toolery.adapters.openai_raw import OpenAIRawAdapter
from toolery.core.models import (
    Budget,
    Category,
    Scenario,
    Scoring,
    ScoringCheck,
    Tier,
    ToolCall,
    ToolResponseRule,
    TraceResult,
)
from toolery.core.runner import Runner, strict_view

URL = "http://localhost:8000/v1/chat/completions"


def _scenario(max_calls: int = 1, max_turns: int = 2) -> Scenario:
    return Scenario(
        id="t-01-slack", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="What's the weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=max_calls, max_turns=max_turns, timeout_seconds=30),
        tool_responses={"get_weather": [
            ToolResponseRule(match="any", returns={"temp_c": 7, "condition": "cloudy"}),
        ]},
        scoring=Scoring(required=[
            ScoringCheck(check="tool_called", tool="get_weather"),
            ScoringCheck(check="response_contains", patterns=["7"]),
        ]),
    )


def _tool_turn(n: int) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": f"c{n}", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"location": "Warsaw"}'}}],
    }}]})


def _answer() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": "It is 7°C and cloudy in Warsaw."}}]})


async def _run(slack: float, script: list[httpx.Response]):
    respx.post(URL).mock(side_effect=script)
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    runner = Runner(adapters={"raw": adapter}, trials=1, budget_slack=slack)
    (result,) = await runner.run([_scenario()])
    return result


# The model double-checks the weather (2 calls, budget 1), then answers.
_OVERRUN = [_tool_turn(1), _tool_turn(2), _answer()]


@pytest.mark.asyncio
@respx.mock
async def test_strict_score_identical_with_and_without_slack():
    strict = await _run(1.0, list(_OVERRUN))
    respx.reset()
    slack = await _run(2.0, list(_OVERRUN))
    for field in ("status", "score", "failure_kind", "call_count"):
        assert getattr(slack, field) == getattr(strict, field), field
    assert strict.status == "fail" and strict.failure_kind == "budget_violated"


@pytest.mark.asyncio
@respx.mock
async def test_correctness_sees_the_finished_run():
    strict = await _run(1.0, list(_OVERRUN))
    respx.reset()
    slack = await _run(2.0, list(_OVERRUN))
    # Cut at the budget, the normal run never gets to answer…
    assert strict.correctness_score == 0.0
    assert strict.trace.final_response is None
    # …with slack it finishes, and the answer was right.
    assert slack.correctness_score == 1.0
    assert "7°C" in slack.trace.final_response
    assert slack.trace.adapter_metadata["budget_slack"] == {"factor": 2.0, "cut": True}


@pytest.mark.asyncio
@respx.mock
async def test_slack_is_a_no_op_for_a_run_within_budget():
    within = [_tool_turn(1), _answer()]
    strict = await _run(1.0, list(within))
    respx.reset()
    slack = await _run(2.0, list(within))
    assert slack.status == strict.status == "pass"
    assert slack.correctness_score == strict.correctness_score == 1.0
    assert slack.trace.adapter_metadata["budget_slack"]["cut"] is False


@pytest.mark.asyncio
async def test_adapter_that_shows_the_budget_never_gets_slack():
    """hermes puts the budget in its prompt — inflating it would change the
    model's behavior and break strict_view's exactness."""
    seen: list[Budget] = []

    class PromptBudgetAdapter(MockAdapter):
        async def run_scenario(self, scenario, model, timeout):
            seen.append(scenario.budget)
            return await super().run_scenario(scenario, model, timeout)

    adapter = PromptBudgetAdapter({"t-01-slack": ScenarioPlan(final_response="7")})
    await Runner(adapters={"mock": adapter}, trials=1, budget_slack=3.0).run([_scenario()])
    assert seen[0].max_tool_calls == 1 and seen[0].max_turns == 2


def _trace(calls: list[tuple[int, str]], final: str | None = "done") -> TraceResult:
    return TraceResult(
        scenario_id="t", adapter="raw", trial_index=0, messages=[],
        tool_calls=[ToolCall(index=turn, name=name) for turn, name in calls],
        final_response=final, started_at_iso="2026-09-12T00:00:00Z", duration_ms=1)


B = Budget(max_tool_calls=2, max_turns=3, timeout_seconds=30)


def test_strict_view_uncut_trace_is_returned_unchanged():
    t = _trace([(0, "a"), (1, "b")])
    assert strict_view(t, B) is t


def test_strict_view_cuts_right_after_the_first_call_over_budget():
    # Two parallel calls on turn 1 push past the budget: the adapter stops
    # right after the 3rd call, mid-turn, with no answer.
    v = strict_view(_trace([(0, "a"), (1, "b"), (1, "c"), (2, "d")]), B)
    assert [c.name for c in v.tool_calls] == ["a", "b", "c"]
    assert v.final_response is None


def test_strict_view_calls_on_the_last_allowed_turn_leave_no_answer():
    # max_turns=3 → requests 0..3; tool calls on request 3 end the run.
    b = Budget(max_tool_calls=10, max_turns=3, timeout_seconds=30)
    v = strict_view(_trace([(0, "a"), (3, "b")], final="late answer"), b)
    assert [c.name for c in v.tool_calls] == ["a", "b"]
    assert v.final_response is None


def test_strict_view_drops_turns_beyond_the_limit_and_their_error():
    b = Budget(max_tool_calls=10, max_turns=1, timeout_seconds=30)
    t = _trace([(0, "a"), (2, "b")]).model_copy(update={"error": "HTTPError: 500"})
    v = strict_view(t, b)
    assert [c.name for c in v.tool_calls] == ["a"]
    assert v.error is None and v.final_response is None


@pytest.mark.asyncio
@respx.mock
async def test_strict_score_identical_when_only_the_turn_limit_binds():
    """The subtle case: 2 sequential calls fit the call budget (5) but not the
    turn limit (1 → requests 0..1). A normal run ends after its 2nd request
    without an answer; with slack the model answers on request 3. Only
    strict_view keeps the strict score from turning that into a pass."""
    def scenario():
        return _scenario(max_calls=5, max_turns=1)

    async def run(slack):
        respx.post(URL).mock(side_effect=[_tool_turn(1), _tool_turn(2), _answer()])
        adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
        (r,) = await Runner(adapters={"raw": adapter}, trials=1,
                            budget_slack=slack).run([scenario()])
        return r

    strict = await run(1.0)
    respx.reset()
    slack = await run(2.0)
    assert strict.status == "fail" and strict.failure_kind != "budget_violated"
    for field in ("status", "score", "failure_kind", "call_count"):
        assert getattr(slack, field) == getattr(strict, field), field
    assert slack.correctness_score == 1.0
