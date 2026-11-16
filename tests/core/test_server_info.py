"""Server facts are best effort: they inform a run, they never break one."""
from __future__ import annotations

import httpx
import respx

from toolery.core.server_info import approx_tokens, count_tokens, fetch_server_info

BASE = "http://localhost:8000/v1"
ROOT = "http://localhost:8000"


@respx.mock
def test_llama_server_props():
    respx.get(f"{ROOT}/props").mock(return_value=httpx.Response(200, json={
        "default_generation_settings": {"n_ctx": 65536}, "total_slots": 4,
        "model_path": "/models/Neural-v1-24B.Q6_K.gguf", "build_info": "b7000",
        "chat_template": "{{ messages }}"}))
    info = fetch_server_info(BASE)
    assert info["backend"] == "llama-server" and info["n_ctx"] == 65536
    assert info["total_slots"] == 4 and info["model_path"] == "Neural-v1-24B.Q6_K.gguf"
    assert len(info["chat_template_sha"]) == 12


@respx.mock
def test_koboldcpp_fallback():
    respx.get(f"{ROOT}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{ROOT}/api/extra/true_max_context_length").mock(
        return_value=httpx.Response(200, json={"value": 32768}))
    respx.get(f"{ROOT}/api/extra/version").mock(
        return_value=httpx.Response(200, json={"result": "KoboldCpp", "version": "1.99"}))
    assert fetch_server_info(BASE) == {"backend": "koboldcpp", "n_ctx": 32768, "build_info": "1.99"}


@respx.mock
def test_unknown_or_unreachable_server_yields_nothing():
    respx.get(f"{ROOT}/props").mock(side_effect=httpx.ConnectError("refused"))
    assert fetch_server_info(BASE) == {}
    respx.post(f"{ROOT}/tokenize").mock(side_effect=httpx.ConnectError("refused"))
    assert count_tokens(BASE, "hello") is None


@respx.mock
def test_token_count_from_either_backend():
    respx.post(f"{ROOT}/tokenize").mock(return_value=httpx.Response(200, json={"tokens": [1, 2, 3]}))
    assert count_tokens(BASE, "a b c") == 3
    respx.post(f"{ROOT}/tokenize").mock(return_value=httpx.Response(404, json={}))
    respx.post(f"{ROOT}/api/extra/tokencount").mock(return_value=httpx.Response(200, json={"value": 5}))
    assert count_tokens(BASE, "a b c") == 5


def test_documented_approximation():
    assert approx_tokens("x" * 4000) == 1001
