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
    mixed = '"Name?" he asks. "Mara," you say. "Mara Venn." He nods. You add, "From the south."'
    assert dialogue_spans(mixed, exclude_player=True) == ["Name?"]


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
    # The clerk asks; the narrator answers for the player. Not a leak.
    ('The clerk dips his pen and looks up. "Name?"\n\n"Mara Venn," you say, and he writes it '
     'down in a careful hand.', "fail", "spoke_for_player"),
    ('"Name?" he asks.\n\nYou answer, "Mara Venn." He nods and writes.', "fail", "spoke_for_player"),
    # ...but a clerk who says a name nobody gave him is one.
    ('"Name?" he asks, then answers himself. "No matter. Mara Venn, the mapmaker."',
     "fail", "knowledge_leak"),
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


# Replies of Neural-v1-24B, first real run (2026-11-16), shortened.
REAL = [
    # The narrator answers for the player; the clerk then repeats what he was told.
    ('"Name?"\n\n"Mara Venn," you say.\n\nThe clerk nods as if the name is familiar. "And where is '
     'it you\'re bound, Mistress Venn?"\n\n"Nowhere. I\'m on my way back to Greywater."',
     "spoke_for_player"),
    # Question and answer exchanged with no narration at all.
    ('"Name?"\n\n"Mara Venn."\n\n"Destination?"\n\n"Greywater, at the coast. I\'m a cartographer."\n\n'
     'The clerk\'s pen stops. He looks at your face. "I know that name. You were the one they were '
     'asking after at the Wick crossing."', "spoke_for_player"),
    # The model slides into the player's first person.
    ('"What name shall I write, ma\'am?"\n\nI pause. The name I have been using has served me well '
     'enough.\n\n"Venn," I say. "Mara Venn."\n\nThe clerk writes it.', "spoke_for_player"),
    ('"Name?"\n\nThe question has been asked in every town between here and the coast.\n\n"The name '
     'is Venn," you say. "Mara Venn." It sounds different, spoken aloud like this.',
     "spoke_for_player"),
    ('"Name?"\n\n"I\'m Mara Venn, a cartographer, and I\'m bound for Greywater on Guild business." I '
     'look at him steadily. "The Pilots\' Guild."\n\nThe clerk does not write my name.',
     "spoke_for_player"),
    ('The clerk dips his pen. "Name and destination, if you please."\n\n"Mara Venn. Bound for '
     'Greywater."\n\nHe writes both down.', "spoke_for_player"),
    ('"Name and destination, if you please." He has a thin face and eyes the colour of slate.\n\n'
     '"Mara Venn. Bound for Greywater."\n\nThe clerk writes both names. "Mara Venn. That name was '
     'mentioned here two months ago."', "spoke_for_player"),
    # The player's own action between the question and her answer.
    ('"The name, please?"\n\nI push the coin across the desk.\n\n"Mara Venn. Going north, to the '
     'river road."\n\nThe clerk writes it in. "Mara Venn. River road."', "spoke_for_player"),
    # "I'm looking for..." is not a self-introduction.
    ('A rider reins in beside the milestone.\n\n"I\'m looking for a mapmaker called Mara Venn," he '
     'says. "Have you seen her?"', "knowledge_leak"),
    ('He stops. "I am looking for a surveyor named Mara Venn. I was told she passed this way."',
     "knowledge_leak"),
    # A stranger who knows.
    ('The man is perhaps sixty, with a seaman\'s beard. He looks you up and down.\n\n"Mara Venn," he '
     'says, and waits.\n\nThe road is empty in both directions.', "knowledge_leak"),
    ('The milestone reads "Seven miles to Stonyford", and you have been following it for hours.\n\n'
     '"Mara Venn," he says. "The cartographer." It is not a question.', "knowledge_leak"),
]


@pytest.mark.parametrize("reply,kind", REAL)
def test_leak_checks_on_real_replies(reply, kind):
    assert _verdict("rp-leak-ledger-8k", reply) == ("fail", kind)


# -------------------------------------------------- retention, player voice

RECALL = [
    ("rp-recall-arm-8k", "Your left forearm, burned by lamp oil at the inn.", "pass", None),
    ("rp-recall-arm-8k", "(OOC: That's right, it was the left one.)", "pass", None),
    ("rp-recall-arm-8k", "It was your right forearm, scalded by the lamp oil.", "fail", "forgot_detail"),
    ("rp-recall-arm-8k", "Your forearm was burned by hot lamp oil.", "fail", "forgot_detail"),
    ("rp-recall-compass-8k", "A small heron and the initials E.V.; your brother Edrin gave it to you.",
     "pass", None),
    ("rp-recall-compass-8k", "A heron and the letters E.V. It was a gift from your father.",
     "fail", "forgot_detail"),
    ("rp-recall-inn-8k", "The Heron’s Rest, kept by Hesper Quill.", "pass", None),
    ("rp-recall-inn-8k", "The Saltmarsh Inn, kept by Hesper.", "fail", "forgot_detail"),
]

VOICE = [
    ('The pie woman waits, peel in hand, while the crowd pushes past her stall. "Take your time, '
     'love, the pies won\'t," she says, and turns to serve a drover.', "pass", None),
    # A character introducing himself is not the player speaking.
    ('The old man leans on his broom and waits a while longer. "I\'m Pask, the verger here," he '
     'says at last. "Take your time."', "pass", None),
    # Conditional narration is not a decision.
    ("If you say nothing, she will serve the next customer; the queue behind you is growing and "
     "the oven is hot.", "pass", None),
    ('The square roars around you. "Mutton, please," you say, counting a penny into her palm.',
     "fail", "spoke_for_player"),
    ('You say, "I\'ll take the mutton," and she wraps it in a dock leaf before you can change '
     'your mind.', "fail", "spoke_for_player"),
    # Forms seen in the first real run.
    ('She waits. "Venn," I say. "Mara Venn."', "fail", "spoke_for_player"),
    ('The pie woman waits.\n\nI pause. The name I have been using has served me well enough, but '
     'it is not mine, and the market is very loud around me.', "fail", "spoke_for_player"),
    ('"Well, love?" she says.\n\n"I\'m Mara Venn, a cartographer, and I would like the mutton."',
     "fail", "spoke_for_player"),
    ("You decide to trust her, and before long she has heard the whole story of the Guild and "
     "the estuary chart.", "fail", "spoke_for_player"),
]


@pytest.mark.parametrize("scenario_id,reply,status,kind", RECALL)
def test_retention_checks(scenario_id, reply, status, kind):
    assert _verdict(scenario_id, reply) == (status, kind)


@pytest.mark.parametrize("reply,status,kind", VOICE)
def test_player_voice_checks(reply, status, kind):
    assert _verdict("rp-player-voice-8k", reply) == (status, kind)


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
