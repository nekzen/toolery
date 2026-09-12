"""The raw adapter's HTTP timeout must follow the scenario's time budget
(--timeout-scale), not the client's fixed 120s — which used to cut any single
turn longer than two minutes, whatever the budget said."""
from __future__ import annotations

import httpx
import pytest
import respx

import toolery.adapters.openai_raw as raw_mod
import toolery.tools.generic  # noqa: F401 — registers get_weather
from toolery.adapters.openai_raw import (
    OpenAIRawAdapter,
    ScenarioTimeBudgetExhausted,
    _request_timeout,
)
from toolery.core.models import (
    Budget,
    Category,
    Scenario,
    Scoring,
    ScoringCheck,
    Tier,
    ToolResponseRule,
)
from toolery.core.scorer import evaluate

URL = "http://localhost:8000/v1/chat/completions"


def _scenario() -> Scenario:
    return Scenario(
        id="t-timeout", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="Weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=3, max_turns=3, timeout_seconds=30),
        tool_responses={"get_weather": [ToolResponseRule(match="any", returns={"temp_c": 7})]},
        scoring=Scoring(required=[ScoringCheck(check="response_contains", patterns=["7"])]),
    )


def _tool_turn() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"location": "Warsaw"}'}}]}}]})


def _answer() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": "7°C"}}]})


@pytest.mark.asyncio
@respx.mock
async def test_request_timeout_follows_the_budget_not_120s():
    route = respx.post(URL).mock(return_value=_answer())
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    await adapter.run_scenario(_scenario(), model="m", timeout=300)
    t = route.calls.last.request.extensions["timeout"]
    assert 290 < t["read"] <= 300      # the scenario budget, not the old 120s cap
    assert t["connect"] == 10.0        # a dead server still fails fast


@pytest.mark.asyncio
@respx.mock
async def test_timeout_keeps_the_partial_trace_and_classifies_as_timeout():
    respx.post(URL).mock(side_effect=[_tool_turn(), httpx.ReadTimeout("slow turn")])
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    scenario = _scenario()
    trace = await adapter.run_scenario(scenario, model="m", timeout=300)
    assert trace.error.startswith("timeout:")
    assert [c.name for c in trace.tool_calls] == ["get_weather"]   # how far it got
    assert evaluate(scenario, trace).failure_kind == "timeout"


def test_later_requests_get_only_the_remaining_budget(monkeypatch):
    monkeypatch.setattr(raw_mod.time, "monotonic", lambda: 1000.0)
    t = _request_timeout(deadline=1050.0)
    assert t.read == 50.0 and t.connect == 10.0
    t = _request_timeout(deadline=1004.0)
    assert t.read == 4.0 and t.connect == 4.0
    with pytest.raises(ScenarioTimeBudgetExhausted):
        _request_timeout(deadline=999.0)
    assert _request_timeout(None) is httpx.USE_CLIENT_DEFAULT
