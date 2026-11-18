"""An LLM judge for the rp_long_context leak and player-voice probes.

The deterministic checks decide who speaks a line from quotes and speech
tags. They miss what needs reading: a letter the narrator makes the player
hand over, a stranger who "guesses" the exact hiding place, a messenger
invented to explain how a character knows. The judge reads the reply and
answers the same question the rules answer, with a quote as proof.

It never runs during a benchmark. `toolery judge` passes over stored traces,
like `rescore`, and the verdicts sit next to the rules', never in their place:
a disagreement is information. The judge must be another model than the one
tested, of another family if possible, run at temperature 0 without
reasoning. Before trusting it, measure its agreement with hand-labelled
replies: `toolery judge --calibrate`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from toolery.core.scorer import _MATCH_CHAR_TRANS

PROMPT_VERSION = "1"
VERDICTS = ("ok", "knowledge_leak", "spoke_for_player")
SRC = Path(__file__).resolve().parents[2] / "fixtures" / "src"
CALIBRATION = Path(__file__).resolve().parents[2] / "fixtures" / "judge" / "calibration.yaml"

_SYSTEM = ("You are a careful, literal reader. You judge one reply written by the narrator of an "
           "interactive roleplay, against the rules the narrator was given. You answer with a "
           "single JSON object and nothing else.")

_LEAK_QUESTION = """Decide what the reply does with the secret described above. Read it in order and report the FIRST of these that happens:
- "spoke_for_player": the narrator writes the player character giving the secret away herself: her words, her answer, her gesture or action, in the second or the first person ("you say", "I say", an answer with no speaker right after a question put to her).
- "knowledge_leak": a character who was never told the secret says it, writes it, reads it out, uses it or goes straight to it. Guessing it exactly counts. So does a character who claims someone told them, since the fact says nobody else knows.
- "ok": neither happens. A character may ask, wait or look; the narrator's own description may mention the secret, as long as no character learns or uses it and the player's words and actions are not written for her."""

_VOICE_QUESTION = """The player has chosen to stay silent for a moment. Does the narrator write anything the player character says, decides, does or thinks?
- "spoke_for_player": yes: a line spoken by her ("you say", "I say", an answer with no speaker), a decision or an action taken for her ("you decide to trust him", "you hand him the letter"), or her thoughts written in the first person.
- "ok": no: the narrator only describes the world, the other characters and what she can see or hear, and leaves her the next move."""

_ANSWER = """Answer with this JSON object only:
{"verdict": "ok" | "knowledge_leak" | "spoke_for_player",
 "quote": "the exact words of the reply that show it, copied character for character; empty when the verdict is ok",
 "reason": "one short sentence"}"""


@dataclass
class Case:
    """One reply to judge, with what the judge needs to read it."""
    scenario_id: str
    probe: str
    story: str
    system: str
    prompt: str
    reply: str


def probe_family(probe: str) -> str | None:
    """The question a probe asks the judge: leak probes and player-voice.
    Card variants and help conditions keep their family."""
    if probe.startswith("leak-"):
        return "leak"
    if probe.startswith("player-voice"):
        return "voice"
    return None


def story_fact(story: str) -> str:
    card = yaml.safe_load((SRC / story / "card.yaml").read_text(encoding="utf-8"))
    return str(card.get("judge_fact") or "").strip()


def build_messages(case: Case) -> list[dict]:
    family = probe_family(case.probe)
    if family is None:
        raise ValueError(f"no judge question for probe {case.probe!r}")
    parts = ["## The rules the narrator was given", case.system.strip()]
    if family == "leak":
        parts += ["## A fact the narrator must respect", story_fact(case.story)]
    parts += ["## The player's last message", case.prompt.strip(),
              "## The narrator's reply to judge", "<<<", case.reply.strip(), ">>>",
              "## Question", _LEAK_QUESTION if family == "leak" else _VOICE_QUESTION,
              _ANSWER]
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_MATCH_CHAR_TRANS)).strip().lower()


def parse_verdict(text: str | None, reply: str) -> dict:
    """The judge's answer, checked. `valid` is False when the JSON is
    unreadable, the verdict unknown, or the quote is not in the reply: a
    verdict that cannot show its proof is not used."""
    out = {"verdict": None, "quote": "", "reason": "", "valid": False, "error": None}
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        out["error"] = "no JSON object"
        return out
    out["verdict"] = data.get("verdict")
    out["quote"] = str(data.get("quote") or "").strip()
    out["reason"] = str(data.get("reason") or "").strip()
    if out["verdict"] not in VERDICTS:
        out["error"] = f"unknown verdict {out['verdict']!r}"
    elif out["verdict"] != "ok" and not out["quote"]:
        out["error"] = "no quote"
    elif out["quote"] and _norm(out["quote"].strip('"“”')) not in _norm(reply):
        out["error"] = "the quote is not in the reply"
    else:
        out["valid"] = True
    return out


class JudgeClient:
    """An OpenAI-compatible endpoint (llama-server, vLLM) used as a judge."""

    def __init__(self, base_url: str, api_key: str = "", model: str = "judge",
                 timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._http = httpx.Client(timeout=timeout)

    def identity(self) -> dict:
        """Name and build of the judge, recorded with every verdict."""
        ident = {"base_url": self.base_url, "model": self.model}
        try:
            props = self._http.get(f"{self.base_url}/props", headers=self.headers).json()
            ident["model_path"] = Path(str(props.get("model_path") or "")).name or None
            ident["build"] = props.get("build_info")
        except (httpx.HTTPError, ValueError):
            pass
        return ident

    def judge(self, case: Case) -> dict:
        body = {"model": self.model, "messages": build_messages(case),
                "temperature": 0, "max_tokens": 400, "seed": 0,
                # Reasoning off: short, reproducible verdicts.
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"}}
        resp = self._http.post(f"{self.base_url}/v1/chat/completions", json=body,
                               headers=self.headers)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"].get("content")
        return parse_verdict(text, case.reply)

    def close(self) -> None:
        self._http.close()


def record(verdict: dict, identity: dict) -> dict:
    """What is stored under meta_json["judge"]."""
    return {**{k: verdict[k] for k in ("verdict", "quote", "reason", "valid", "error")},
            "judge": identity, "prompt_version": PROMPT_VERSION,
            "at": datetime.now(UTC).isoformat(timespec="seconds")}


def rule_label(status: str, failure_kind: str | None) -> str | None:
    """The rules' answer in the judge's terms; None when the rules gave none
    (no dialogue, empty reply, an error)."""
    if status == "pass":
        return "ok"
    if failure_kind in ("knowledge_leak", "spoke_for_player"):
        return failure_kind
    return None


def load_calibration(path: Path = CALIBRATION) -> list[dict]:
    """Hand-labelled replies: scenario id, reply, expected verdict."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
