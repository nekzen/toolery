"""`toolery run --text-tool-calls`: a measurement mode for models that write
their tool calls as JSON in the reply text. The calls are run, and the
results are stored under the adapter name "raw-text", apart from raw runs."""
from __future__ import annotations

import json
import sqlite3

import httpx
import respx
from typer.testing import CliRunner

from toolery.cli import app

BASE = "http://llama.test:8080"
SID = "code-review-medium-01-hardcoded-secret"


def _reply(request: httpx.Request) -> httpx.Response:
    messages = json.loads(request.content)["messages"]
    if messages[-1]["role"] == "tool":
        text = "Hardcoded secret in src/config.py: move it to an environment variable."
    else:   # what Neural-v1-24B writes, in place of [TOOL_CALLS]
        text = ('Let me read it.\n\n```python\n{"name": "read_file", '
                '"arguments": "{\\"path\\": \\"src/config.py\\"}"}\n```')
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 40}})


@respx.mock
def test_text_tool_calls_run_is_stored_as_raw_text(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path / "results"))
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(200, json={
        "default_generation_settings": {"n_ctx": 65536}}))
    respx.post(f"{BASE}/tokenize").mock(return_value=httpx.Response(200, json={"tokens": [0]}))
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_reply)
    result = CliRunner().invoke(app, ["run", "--model", "neural-text", "--ids", SID, "--trials", "1",
                                      "--concurrency", "1", "--text-tool-calls", "--base-url", BASE])
    assert result.exit_code == 0, result.output
    db = sqlite3.connect(tmp_path / "results" / "runs.db")
    (adapter, calls), = db.execute("select adapter, call_count from scenario_results").fetchall()
    assert (adapter, calls) == ("raw-text", 1)
    (cfg,), = db.execute("select config_json from runs").fetchall()
    assert json.loads(cfg)["text_tool_calls"] is True
