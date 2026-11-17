"""The rp_long_context category end to end: `toolery run` against a
llama-server stand-in (HTTP mocked at the wire), then `toolery rp-report`.

The stand-in behaves like a model with the defects the category measures: it
leaks the player's name on some seeds, and absorbs "Pause scenario." once the
context is long.
"""
from __future__ import annotations

import json

import httpx
import respx
from typer.testing import CliRunner

from toolery.cli import app

BASE = "http://llama.test:8080"


def _reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    probe = body["messages"][-1]["content"]
    long_context = len(body["messages"]) > 60
    seed = body["seed"]
    assert body["temperature"] == 0.8 and body["min_p"] == 0.05 and body["max_tokens"] in (350, 500)
    assert body["messages"][0]["role"] == "system" and "Harbor Road" in body["messages"][0]["content"]
    if "milestone" in probe or "ledger" in probe:
        text = ('The man stops and looks you over for a while before he speaks.\n\n'
                + ('"Mara Venn, as I live," he says.' if seed % 2 else '"What do they call you, traveller?" he asks.'))
    elif "Hesper?" in probe:
        text = 'Hesper looks up from the dough and laughs. "Mara. As if I would forget," she says.'
    elif long_context and probe.startswith("Pause"):
        text = 'The old man blinks. "Pause what?" He leans on his broom and waits for an answer.'
    else:
        text = "Paused. Let me know when you would like to resume."
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 15000 if long_context else 8000, "completion_tokens": 40}})


@respx.mock
def test_run_then_report(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(200, json={
        "default_generation_settings": {"n_ctx": 65536}, "total_slots": 1,
        "model_path": "/models/Neural-v1-24B.gguf"}))
    respx.post(f"{BASE}/tokenize").mock(
        side_effect=lambda req: httpx.Response(200, json={
            "tokens": [0] * (len(json.loads(req.content)["content"]) // 4)}))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_reply)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--model", "neural-test", "--category", "rp_long_context",
                                 "--trials", "4", "--concurrency", "1", "--base-url", BASE])
    assert result.exit_code == 0, result.output
    assert chat.call_count == (46 + 4 * 6) * 4   # repetition: six replies per trial

    report = runner.invoke(app, ["rp-report", "--json"])
    assert report.exit_code == 0, report.output
    cells = {(c["probe"], c["length"]): c for c in json.loads(report.output)}
    assert len(cells) == 50
    leak = cells[("leak-stranger", "16k")]
    assert (leak["trials"], leak["passed"], leak["failures"]) == (4, 2, {"knowledge_leak": 2})
    assert {c["story"] for c in cells.values()} == {"harbor-road"}
    assert leak["prompt_tokens"] == 15000 and leak["sampling"] == {
        "temperature": 0.8, "min_p": 0.05, "max_tokens": 350}
    assert leak["examples"][0]["proof"] == 'spoken: "Mara Venn, as I live,"'
    assert leak["examples"][0]["seed"] in (1001, 1003)
    assert cells[("control-known", "8k")]["pass_rate"] == 1.0
    assert cells[("ooc-pause", "8k")]["pass_rate"] == 1.0
    assert cells[("ooc-pause", "16k")]["failures"] == {"ooc_absorbed": 4}
    assert cells[("ooc-stop", "16k")]["pass_rate"] == 1.0
    # The stand-in gives the same reply six times: as repetitive as it gets.
    rep = cells[("repetition", "8k")]
    assert rep["failures"] == {"repetitive": 4}
    assert rep["metrics"]["ngram_share"] == 1.0 and rep["metrics"]["same_opener"] == 6
    # 40 completion tokens per reply against a cap of 500: nothing was cut.
    assert (rep["metrics"]["truncated"], rep["metrics"]["replies"]) == (0, 24)

    monkeypatch.setenv("COLUMNS", "220")     # four context lengths side by side
    table = runner.invoke(app, ["rp-report", "--failures"])
    assert table.exit_code == 0, table.output
    assert "leak-stranger" in table.output and "knowledge_leak 2" in table.output
    assert "8k (~8000 tok)" in table.output and "16k (~15000 tok)" in table.output
    assert "self-repetition" in table.output and "100.0%" in table.output
    assert "cut replies" in table.output and "0/24" in table.output
    assert "harbor-road" in table.output


@respx.mock
def test_a_conversation_too_long_for_the_server_is_skipped_not_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(200, json={
        "default_generation_settings": {"n_ctx": 10000}}))  # only the 8k conversation fits
    respx.post(f"{BASE}/tokenize").mock(
        side_effect=lambda req: httpx.Response(200, json={
            "tokens": [0] * (len(json.loads(req.content)["content"]) // 4)}))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_reply)
    result = CliRunner().invoke(app, ["run", "--model", "small-ctx", "--category", "rp_long_context",
                                      "--trials", "1", "--concurrency", "1", "--base-url", BASE])
    assert result.exit_code == 0, result.output
    # The ten single-reply 8k probes fit. rp-repetition-8k does not: six
    # replies of up to 500 tokens on top of ~7000 of conversation.
    assert "Skipping 40 scenario(s)" in result.output and "rp-repetition-8k" in result.output
    assert chat.call_count == 10


@respx.mock
def test_a_server_that_wants_a_key_stops_the_run_before_any_trial(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(401, json={"error": "Invalid API Key"}))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_reply)
    result = CliRunner().invoke(app, ["run", "--model", "locked", "--category", "rp_long_context",
                                      "--trials", "1", "--base-url", BASE])
    assert result.exit_code == 2 and "OPENAI_API_KEY" in result.output
    assert chat.call_count == 0


def test_a_refused_key_is_not_a_model_crash():
    from toolery.core.scorer import _classify_error
    assert _classify_error("HTTP 401: Invalid API Key") == "auth_error"
    assert _classify_error("segfault in sampler") == "model_crash"
