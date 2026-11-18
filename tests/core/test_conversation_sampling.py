"""Frozen conversations, per-scenario sampling, per-trial metadata and
declared failure kinds — the foundations of the long-context / RP category."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime

import httpx
import pytest
import respx
import yaml

from toolery.adapters.openai_raw import OpenAIRawAdapter
from toolery.core.models import ToolCall
from toolery.core.runner import Runner
from toolery.core.scenario import load_scenario
from toolery.core.scorer import evaluate
from toolery.core.store import Store

URL = "http://localhost:8000/v1/chat/completions"
FIXTURE = {
    "name": "demo", "length": "8k", "approx_tokens": 42, "system": "You are the narrator.",
    "messages": [{"role": "user", "content": "I walk in."},
                 {"role": "assistant", "content": "The inn is warm."}],
}


def _write(tmp_path, *, sampling=None, scoring=None, conversation=True):
    (tmp_path / "fx.json").write_text(json.dumps(FIXTURE), encoding="utf-8")
    data = {
        "id": "t-01-conv", "title": "t", "tier": "easy", "category": "creative_writing",
        "domain": "generic", "description": "d", "prompt": "I greet the stranger.",
        "tools": [], "budget": {"max_tool_calls": 0, "max_turns": 1, "timeout_seconds": 30},
        "scoring": scoring or {"required": [{"check": "response_contains", "patterns": ["hello"]}]},
    }
    if conversation:
        data["conversation"] = {"fixture": "fx.json"}
    if sampling:
        data["sampling"] = sampling
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_scenario(path)


def _answer(text="hello there"):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 3}})


def test_loader_resolves_the_fixture(tmp_path):
    s = _write(tmp_path)
    assert [m.content for m in s.history] == ["I walk in.", "The inn is warm."]
    assert s.system_prompt == "You are the narrator."
    c = s.conversation
    assert (c.name, c.length, c.approx_tokens, c.n_messages) == ("demo", "8k", 42, 2)
    canonical = json.dumps(FIXTURE, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert c.sha256 == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Re-indenting or CRLF line endings do not change the identity.
    (tmp_path / "fx.json").write_bytes(json.dumps(FIXTURE, indent=2).replace("\n", "\r\n").encode())
    assert load_scenario(tmp_path / "s.yaml").conversation.sha256 == c.sha256


@pytest.mark.asyncio
@respx.mock
async def test_adapter_sends_history_and_sampling_but_traces_a_marker(tmp_path):
    route = respx.post(URL).mock(return_value=_answer())
    s = _write(tmp_path, sampling={"temperature": 0.8, "min_p": 0.05, "seed": 7, "max_tokens": 300})
    trace = await OpenAIRawAdapter(base_url="http://localhost:8000").run_scenario(s, "m", 30)
    body = json.loads(route.calls.last.request.content)
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
    assert body["messages"][-1]["content"] == "I greet the stranger."
    assert (body["temperature"], body["min_p"], body["seed"], body["max_tokens"]) == (0.8, 0.05, 7, 300)
    # The trace identifies the conversation instead of copying it.
    assert [m.role for m in trace.messages] == ["system", "system", "user", "assistant"]
    assert "2 messages omitted" in trace.messages[1].content
    assert trace.adapter_metadata["sampling"]["seed"] == 7
    assert trace.adapter_metadata["conversation"]["sha256"] == s.conversation.sha256


@pytest.mark.asyncio
@respx.mock
async def test_scenarios_without_sampling_are_unchanged(tmp_path):
    route = respx.post(URL).mock(return_value=_answer())
    s = _write(tmp_path, conversation=False)
    trace = await OpenAIRawAdapter(base_url="http://localhost:8000").run_scenario(s, "m", 30)
    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.0 and "seed" not in body and "min_p" not in body
    assert "sampling" not in trace.adapter_metadata and "conversation" not in trace.adapter_metadata


@pytest.mark.asyncio
@respx.mock
async def test_each_trial_gets_its_own_reproducible_seed(tmp_path):
    route = respx.post(URL).mock(return_value=_answer())
    s = _write(tmp_path, sampling={"temperature": 0.8, "seed": 100})
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000")
    results = await Runner(adapters={"raw": adapter}, trials=3, concurrency=1).run([s])
    assert sorted(json.loads(c.request.content)["seed"] for c in route.calls) == [100, 101, 102]
    assert sorted(r.trace.adapter_metadata["sampling"]["seed"] for r in results) == [100, 101, 102]


@pytest.mark.asyncio
@respx.mock
async def test_meta_json_is_stored_and_old_databases_migrate(tmp_path):
    respx.post(URL).mock(return_value=_answer())
    s = _write(tmp_path, sampling={"temperature": 0.8, "seed": 5})
    (result,) = await Runner(adapters={"raw": OpenAIRawAdapter(base_url="http://localhost:8000")},
                             trials=1).run([s])
    db = tmp_path / "runs.db"
    store = Store(db)
    store.init_schema()
    with sqlite3.connect(db) as c:      # simulate a database from before the column
        c.execute("ALTER TABLE scenario_results DROP COLUMN meta_json")
    store.init_schema()
    store.create_run(run_id="r", model="m", base_url="x", started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.write_scenario_result("r", result, tags=[], ranking_dims=["overall"], scenario_hash="h",
                                category="creative_writing", tier="easy", trace_path="t.json")
    meta = json.loads(store.fetch_results_for_run("r")[0]["meta_json"])
    assert meta["sampling"] == {"temperature": 0.8, "seed": 5}
    assert meta["conversation"]["length"] == "8k"


def _eval(tmp_path, scoring, response="nothing useful", calls=()):
    from toolery.core.models import Message, TraceResult
    s = _write(tmp_path, scoring=scoring, conversation=False)
    trace = TraceResult(scenario_id=s.id, adapter="raw", trial_index=0,
                        messages=[Message(role="user", content="q")],
                        tool_calls=[ToolCall(index=0, name=n) for n in calls],
                        final_response=response, started_at_iso="2026-11-16T00:00:00Z", duration_ms=1)
    return evaluate(s, trace)


def test_a_check_can_name_the_failure_it_detects(tmp_path):
    leak = {"required": [{"check": "response_contains", "patterns": ["nothing"]}],
            "forbidden": [{"check": "response_contains", "patterns": ["useful"],
                           "failure_kind": "knowledge_leak"}]}
    assert _eval(tmp_path, leak).failure_kind == "knowledge_leak"
    ooc = {"required": [{"check": "response_contains", "patterns": ["paused"],
                         "failure_kind": "ooc_ignored"}]}
    assert _eval(tmp_path, ooc).failure_kind == "ooc_ignored"


def test_a_wrong_answer_is_no_longer_labelled_wrong_tool(tmp_path):
    said = {"required": [{"check": "response_contains", "patterns": ["hello"]}]}
    assert _eval(tmp_path, said).failure_kind == "wrong_answer"
    plain_forbidden = {"required": [{"check": "response_contains", "patterns": ["nothing"]}],
                       "forbidden": [{"check": "response_contains", "patterns": ["useful"]}]}
    assert _eval(tmp_path, plain_forbidden).failure_kind == "forbidden_action"


@pytest.mark.asyncio
@respx.mock
async def test_follow_ups_are_sent_one_by_one_and_replies_joined(tmp_path):
    from toolery.core.models import TURN_SEPARATOR
    answers = iter(["first", "second", "third"])
    route = respx.post(URL).mock(side_effect=lambda req: _answer(next(answers)))
    s = _write(tmp_path).model_copy(update={"follow_ups": ["and then?", "and after?"]})
    trace = await OpenAIRawAdapter(base_url="http://localhost:8000").run_scenario(s, "m", 30)
    assert route.call_count == 3
    last = json.loads(route.calls.last.request.content)["messages"]
    # The third request carries the model's own two earlier replies.
    assert [m["content"] for m in last[-5:]] == [
        "I greet the stranger.", "first", "and then?", "second", "and after?"]
    assert trace.final_response == TURN_SEPARATOR.join(["first", "second", "third"])
    assert trace.error is None
