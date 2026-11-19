"""The LLM judge: prompt, verdict checking, calibration set, CLI."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from toolery.cli import app
from toolery.core.judge import (
    Case,
    build_messages,
    load_calibration,
    parse_verdict,
    probe_family,
    rule_label,
)
from toolery.core.models import Message, TraceResult
from toolery.core.scenario import load_scenario
from toolery.core.scorer import evaluate

RP = Path(__file__).resolve().parents[2] / "scenarios" / "rp_long_context"
REPLY = 'The clerk dips his pen. "Name?"\n\n“Mara Venn,” you say, and he writes it down.'


def test_a_verdict_must_quote_the_reply():
    ok = parse_verdict('{"verdict": "spoke_for_player", "quote": "\\"Mara Venn,\\" you say", '
                       '"reason": "the narrator answers"}', REPLY)
    assert ok["valid"] and ok["verdict"] == "spoke_for_player"     # curly quotes folded
    made_up = parse_verdict('{"verdict": "knowledge_leak", "quote": "You are Mara Venn"}', REPLY)
    assert not made_up["valid"] and made_up["error"] == "the quote is not in the reply"
    assert parse_verdict('{"verdict": "knowledge_leak", "quote": ""}', REPLY)["error"] == "no quote"
    assert parse_verdict('{"verdict": "maybe", "quote": "x"}', REPLY)["error"].startswith("unknown")
    assert parse_verdict("I think it is a leak.", REPLY)["error"] == "no JSON object"
    assert parse_verdict('```json\n{"verdict": "ok", "quote": "", "reason": "r"}\n```', REPLY)["valid"]


def test_the_prompt_carries_the_fact_for_leaks_only():
    s = load_scenario(RP / "rp-leak-ledger-16k.yaml")
    leak = build_messages(Case(s.id, "leak-ledger", "harbor-road", s.system_prompt, s.prompt, REPLY))
    assert "Hesper Quill" in leak[1]["content"] and REPLY.strip() in leak[1]["content"]
    v = load_scenario(RP / "rp-player-voice-16k.yaml")
    voice = build_messages(Case(v.id, "player-voice", "harbor-road", v.system_prompt, v.prompt, "x"))
    assert "Hesper Quill" not in voice[1]["content"].split("## The player's last message")[0] \
        .split("## The rules the narrator was given")[0]
    assert "## A fact the narrator must respect" not in voice[1]["content"]
    snow = build_messages(Case("s", "leak-stranger-secretcard", "snowline", "rules", "p", "r"))
    assert "Ilse" in snow[1]["content"]
    given = build_messages(Case("s", "leak-ledger", "elsewhere", "rules", "p", "r",
                                fact="Only Rook knows the vault code."))
    assert "Only Rook knows the vault code." in given[1]["content"]
    with pytest.raises(ValueError):
        build_messages(Case("s", "ooc-pause", "harbor-road", "", "", ""))


def test_families_and_rule_labels():
    assert probe_family("leak-stranger-told") == "leak"
    assert probe_family("player-voice-reminder") == "voice"
    assert probe_family("recall-arm") is None
    assert rule_label("pass", None) == "ok"
    assert rule_label("fail", "knowledge_leak") == "knowledge_leak"
    assert rule_label("fail", "no_dialogue") is None


def test_the_calibration_labels_are_what_the_rules_say():
    """The hand labels were checked against the rules when they were written;
    if a rule changes, this says which label to look at again."""
    cases = load_calibration()
    assert len(cases) >= 50 and {c["label"] for c in cases} == {
        "ok", "knowledge_leak", "spoke_for_player"}
    for c in cases:
        s = load_scenario(RP / f"{c['scenario']}.yaml")
        r = evaluate(s, TraceResult(scenario_id=s.id, adapter="raw", trial_index=0,
                                    messages=[Message(role="user", content=s.prompt)],
                                    tool_calls=[], final_response=c["reply"],
                                    started_at_iso="2026-11-18T00:00:00Z", duration_ms=1))
        # `rules:` marks a known disagreement: a case the rules get wrong.
        assert rule_label(r.status, r.failure_kind) == c.get("rules", c["label"]), c["reply"][:80]


@respx.mock
def test_calibrate_against_a_stand_in_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    base = "http://judge.test:8001"
    respx.get(f"{base}/props").mock(return_value=httpx.Response(200, json={
        "model_path": "/m/Qwen3.8-27B.gguf", "build_info": "b1"}))

    def answer(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        verdict = {"verdict": "ok", "quote": "", "reason": "a stand-in that sees nothing"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(verdict)}}]})

    route = respx.post(f"{base}/v1/chat/completions").mock(side_effect=answer)
    result = CliRunner().invoke(app, ["judge", "--calibrate", "--base-url", base, "--limit", "4"])
    assert result.exit_code == 0, result.output
    assert route.call_count == 4
    assert "Qwen3.8-27B.gguf" in result.output and "judge vs hand labels" in result.output


@respx.mock
def test_judge_a_run_and_store_the_verdict(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from toolery.core.store import Store

    results = tmp_path / "results"
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(results))
    s = load_scenario(RP / "rp-leak-ledger-16k.yaml")
    trace = TraceResult(scenario_id=s.id, adapter="raw", trial_index=0,
                        messages=[Message(role="user", content=s.prompt)], tool_calls=[],
                        final_response=REPLY, started_at_iso="2026-11-18T00:00:00Z", duration_ms=1,
                        adapter_metadata={"sampling": {"temperature": 0.8, "seed": 1000}})
    store = Store(results / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="m", base_url="x",
                     started_at=datetime.now(UTC).isoformat(), config_json="{}", scenarios_hash="h")
    (results / "runs" / "r1" / "traces").mkdir(parents=True)
    (results / "runs" / "r1" / "traces" / "t.json").write_text(trace.model_dump_json())
    scored = evaluate(s, trace)
    store.write_scenario_result("r1", scored, tags=[], ranking_dims=[], scenario_hash="",
                                category="rp_long_context", tier="hard", trace_path="traces/t.json")
    base = "http://judge.test:8001"
    respx.get(f"{base}/props").mock(return_value=httpx.Response(200, json={"build_info": "b1"}))
    verdict = {"verdict": "spoke_for_player", "quote": "Mara Venn,” you say", "reason": "r"}
    respx.post(f"{base}/v1/chat/completions").mock(return_value=httpx.Response(
        200, json={"choices": [{"message": {"content": json.dumps(verdict)}}]}))
    result = CliRunner().invoke(app, ["judge", "--run", "r1", "--base-url", base, "--write"])
    assert result.exit_code == 0, result.output
    (row,) = store.fetch_results_for_run("r1")
    meta = json.loads(row["meta_json"])
    assert meta["judge"]["verdict"] == "spoke_for_player" and meta["judge"]["valid"]
    assert meta["judge"]["judge"]["build"] == "b1" and meta["judge"]["prompt_version"] == "1"
    assert "sampling" in meta                          # the other keys are kept
