"""The rp_long_context category: its checkers against realistic replies (the
calibration set of the deterministic checks), and its generated files."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from toolery.core.models import Message, TraceResult
from toolery.core.scenario import load_scenario
from toolery.core.scorer import dialogue_spans, evaluate
from toolery.rankings.compute import _result_matches_dimension

ROOT = Path(__file__).resolve().parents[2]
RP = ROOT / "scenarios" / "rp_long_context"

_spec = importlib.util.spec_from_file_location("build_rp_fixtures", ROOT / "scripts" / "build_rp_fixtures.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def _verdict(scenario_id: str, reply: str) -> tuple[str, str | None]:
    s = load_scenario(RP / f"{scenario_id}.yaml")
    trace = TraceResult(scenario_id=s.id, adapter="raw", trial_index=0,
                        messages=[Message(role="user", content=s.prompt)], tool_calls=[],
                        final_response=reply, started_at_iso="2026-11-16T00:00:00Z", duration_ms=1)
    r = evaluate(s, trace)
    return r.status, r.failure_kind


# --------------------------------------------------------------- dialogue

def test_dialogue_is_what_stands_between_double_quotes():
    text = 'Mara waits. "Who are you?" he asks. “Nobody,” she says.\n"Cut off mid-sen'
    assert dialogue_spans(text) == ["Who are you?", "Nobody,", "Cut off mid-sen"]
    assert dialogue_spans("No one speaks. It's quiet.") == []


# ------------------------------------------------------------------- leak

LEAK = [
    # The stranger asks: he does not know.
    ('The man is broad and sunburnt, with a pedlar\'s pack on his shoulders. He stops a few paces '
     'off and looks you over.\n\n"Long way from anywhere to be sitting alone," he says. "What do '
     'they call you, traveller?"', "pass", None),
    # The name in narration is a point-of-view slip, not something he knows.
    ('The man studies you. Mara is a stranger to him, and it shows in his face.\n\n"Lost, are '
     'you?" he asks, leaning on his staff.', "pass", None),
    ('The man stops and smiles as though he had expected to find you here.\n\n"Mara Venn, isn\'t '
     'it?" he says. "I\'ve been looking for you."', "fail", "knowledge_leak"),
    ('He tips his hat. “You must be Mara,” he says warmly. “The road has been talking about you.”',
     "fail", "knowledge_leak"),
    # Cut by max_tokens in the middle of the line.
    ('He stops in the road and squints against the sun, one hand on the strap of his pack.\n\n'
     '"Well now," he says. "If it isn\'t Mara Venn, the mapmaker every', "fail", "knowledge_leak"),
    ('He dips the pen and writes Mara Venn in a neat clerk\'s hand before you have said a word. '
     '"Two pence," he says.', "fail", "knowledge_leak"),
    ("The man stops, looks at you for a long moment, and then walks on down the road without a "
     "word, his boots raising dust in the still afternoon air.", "fail", "no_dialogue"),
]


@pytest.mark.parametrize("reply,status,kind", LEAK)
def test_leak_checks(reply, status, kind):
    assert _verdict("rp-leak-stranger-8k", reply) == (status, kind)


def test_control_requires_the_name_to_be_spoken():
    good = ('Hesper turns from the hearth, flour to the elbows, and looks at you for a long '
            'moment.\n\n"Mara," she says. "Did you think I\'d forget?"')
    assert _verdict("rp-control-known-8k", good) == ("pass", None)
    vague = ('Hesper wipes her hands on her apron and smiles. "Of course I do, child," she says. '
             '"You\'re the mapmaker who spilled my lamp."')
    assert _verdict("rp-control-known-8k", vague) == ("fail", "forgot_name")


# -------------------------------------------------------------------- OOC

OOC = [
    ("Scenario paused. Let me know whenever you'd like to resume, or if you have any questions "
     "about the story so far.", "pass", None),
    ("Of course — stepping out of the story. What's your question?", "pass", None),
    ("(OOC: Sure, pausing here. What would you like to ask?)", "pass", None),
    # Markdown bold is not an asterisk action.
    ("**Scenario paused.** Let me know when you want to continue.", "pass", None),
    # A plain reply may recap, quote the instruction, and name characters.
    ('Understood, I\'ve stopped the roleplay. We left off with Pask asking which you are, a '
     'smuggler or a keeper. Just say "resume" when you want to go on.', "pass", None),
    # Absorbed: a character reacts to the instruction.
    ('The old man blinks at you. "Pause what, now?" He leans on his broom. "I asked you a plain '
     'question, traveller."', "fail", "ooc_absorbed"),
    ('"Pause scenario?" Pask repeats, scratching his chin. "Is that some Guild word?"',
     "fail", "ooc_absorbed"),
    # Absorbed: the story simply goes on.
    ("You tell him you are neither. Pask laughs and waves you up the steps toward the Guild "
     "hall, where the green dome catches the morning light.", "fail", "ooc_absorbed"),
    # Played: the pause happens inside the story.
    ("The scene freezes. The waves hang at the foot of the cliff, the broom stops mid-sweep, and "
     "the world holds its breath until you are ready to go on.", "fail", "ooc_played_pause"),
    ("*The narrator sets down the pen and waits patiently.*", "fail", "ooc_played_pause"),
    # Neither narration marker nor acknowledgment: still not an answer.
    ("The tide comes in behind, covering the shingle one grey fold at a time.",
     "fail", "ooc_no_acknowledgment"),
]


@pytest.mark.parametrize("reply,status,kind", OOC)
def test_ooc_checks(reply, status, kind):
    assert _verdict("rp-ooc-pause-16k", reply) == (status, kind)


# -------------------------------------------------------------- generated

def test_generated_files_match_their_source():
    assert builder.stale(builder.build_all()) == [], \
        "run: uv run python scripts/build_rp_fixtures.py"


def test_every_length_shares_the_same_beginning():
    short, long = (load_scenario(RP / f"rp-ooc-pause-{n}.yaml") for n in ("8k", "16k"))
    assert long.history[:len(short.history)] == short.history
    assert short.conversation.approx_tokens <= 7000 < long.conversation.approx_tokens <= 14000
    assert "Mara Venn" in short.history[4].content        # told to Hesper, early
    assert short.history[-1].role == "assistant"


def test_the_builder_refuses_a_story_that_gives_the_name_away(tmp_path):
    (tmp_path / "card.yaml").write_text(
        "name: t\nname_patterns: ['\\bMara\\b']\nnpc_names: [Bo]\nsystem: s\n", encoding="utf-8")
    (tmp_path / "01-a.yaml").write_text(
        "turns:\n  - user: I wave.\n    narrator: '\"Morning, Mara,\" says Bo.'\n", encoding="utf-8")
    with pytest.raises(builder.BuildError, match="name_allowed"):
        builder.load_story(tmp_path)


def test_cut_keeps_whole_exchanges_and_skips_buckets_it_cannot_fill():
    msgs = [{"role": r, "content": "x" * 400} for r in ("user", "assistant")] * 10   # ~2000 tokens
    assert len(builder.cut("", msgs, 1000)) == 8
    assert len(builder.cut("", msgs, 2500)) == 20
    assert builder.cut("", msgs, 8000) is None


def test_the_category_stays_out_of_rankings():
    assert not _result_matches_dimension("overall", [], "rp_long_context")
    assert _result_matches_dimension("overall", [], "coding")
