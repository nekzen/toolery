"""Best-effort facts about the model server a run talks to.

The same weights score differently under different serving setups, and a
long-context scenario is meaningless if the conversation does not fit the
server's context window. Both llama-server and KoboldCPP expose what is
needed; anything else simply yields no information and the run proceeds.

Nothing here ever raises: every network failure returns None / {}.
"""

from __future__ import annotations

import hashlib
import re

import httpx

_TIMEOUT_S = 4.0
# Documented approximation used when the server cannot count tokens: English
# prose averages about 4 characters per token across common tokenizers.
CHARS_PER_TOKEN = 4.0


def _root(base_url: str) -> str:
    """Server root: the base URL without a trailing /vN API segment."""
    return re.sub(r"/v\d+$", "", base_url.rstrip("/"))


def approx_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def fetch_server_info(base_url: str, api_key: str = "") -> dict:
    """Context size, slots and build of the server, when it says.

    llama-server: GET /props. KoboldCPP: GET /api/extra/true_max_context_length
    and /api/extra/version. Returns {} when neither answers, and
    {"auth_error": status} when the server refuses the key (or its absence).
    """
    root = _root(base_url)
    try:
        with httpx.Client(timeout=_TIMEOUT_S, headers=_headers(api_key)) as client:
            r = client.get(f"{root}/props")
            if r.status_code in (401, 403):
                return {"auth_error": r.status_code}
            if r.status_code == 200 and isinstance(r.json(), dict):
                props = r.json()
                gen = props.get("default_generation_settings") or {}
                template = props.get("chat_template")
                info = {
                    "backend": "llama-server",
                    "n_ctx": gen.get("n_ctx"),
                    "total_slots": props.get("total_slots"),
                    "model_path": str(props.get("model_path") or "").replace("\\", "/").rsplit("/", 1)[-1] or None,
                    "build_info": props.get("build_info"),
                    "chat_template_sha": (hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
                                          if isinstance(template, str) else None),
                }
                return {k: v for k, v in info.items() if v is not None}
            r = client.get(f"{root}/api/extra/true_max_context_length")
            if r.status_code == 200 and isinstance(r.json(), dict) and "value" in r.json():
                info = {"backend": "koboldcpp", "n_ctx": r.json()["value"]}
                v = client.get(f"{root}/api/extra/version")
                if v.status_code == 200 and isinstance(v.json(), dict):
                    info["build_info"] = v.json().get("version")
                return info
    except Exception:
        pass
    return {}


def count_tokens(base_url: str, text: str, api_key: str = "") -> int | None:
    """Exact token count from the server's own tokenizer, or None.

    llama-server: POST /tokenize. KoboldCPP: POST /api/extra/tokencount.
    """
    root = _root(base_url)
    try:
        with httpx.Client(timeout=30.0, headers=_headers(api_key)) as client:
            r = client.post(f"{root}/tokenize", json={"content": text})
            if r.status_code == 200 and isinstance(r.json().get("tokens"), list):
                return len(r.json()["tokens"])
            r = client.post(f"{root}/api/extra/tokencount", json={"prompt": text})
            if r.status_code == 200 and isinstance(r.json().get("value"), int):
                return r.json()["value"]
    except Exception:
        pass
    return None
