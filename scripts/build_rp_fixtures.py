"""Build the frozen conversations and scenarios of the rp_long_context category.

    uv run python scripts/build_rp_fixtures.py           # write
    uv run python scripts/build_rp_fixtures.py --check   # exit 1 if out of date

Source (edit this): fixtures/src/<story>/card.yaml, NN-*.yaml chapters, and
probes.yaml. Output (generated, committed): scenarios/rp_long_context/.

A conversation is cut to each context length by whole exchanges, from the
start: every length shares the same beginning, so a fact given in the first
exchange lies deeper and deeper. Lengths are approximate tokens (characters /
4, system prompt included, see toolery.core.server_info.approx_tokens); each
bucket keeps ~12% of its nominal size free for the probe, the reply and
tokenizer differences. The exact count is the server's `prompt_tokens`,
stored with every result.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

from toolery.core.server_info import approx_tokens

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "fixtures" / "src"
OUT = ROOT / "scenarios" / "rp_long_context"

# nominal length -> (approx-token budget for system + history, timeout s)
BUCKETS = {"8k": (7000, 120), "16k": (14000, 180), "32k": (28000, 300), "64k": (56000, 600)}
MIN_FILL = 0.75   # a bucket the story cannot fill this far is not built


class BuildError(ValueError):
    pass


def load_story(story_dir: Path) -> tuple[dict, list[dict]]:
    card = yaml.safe_load((story_dir / "card.yaml").read_text(encoding="utf-8"))
    name_rx = [re.compile(p) for p in card["name_patterns"]]
    messages: list[dict] = []
    for path in sorted(story_dir.glob("[0-9][0-9]-*.yaml")):
        chapter = yaml.safe_load(path.read_text(encoding="utf-8"))
        for i, turn in enumerate(chapter["turns"]):
            user, narrator = turn["user"].strip(), turn["narrator"].strip()
            if not chapter.get("name_allowed"):
                for rx in name_rx:
                    if rx.search(user) or rx.search(narrator):
                        raise BuildError(
                            f"{path.name} turn {i + 1}: {rx.pattern} appears outside a "
                            "`name_allowed` chapter — the name would no longer be a secret")
            messages += [{"role": "user", "content": user},
                         {"role": "assistant", "content": narrator}]
    return card, messages


def _tokens(system: str, messages: list[dict]) -> int:
    return approx_tokens(system + "".join(m["content"] for m in messages))


def cut(system: str, messages: list[dict], budget: int) -> list[dict] | None:
    """Longest prefix of whole exchanges within `budget`, or None when the
    story is too short to fill the bucket."""
    kept: list[dict] = []
    for i in range(0, len(messages), 2):
        if _tokens(system, kept + messages[i:i + 2]) > budget:
            return kept
        kept = kept + messages[i:i + 2]
    return kept if _tokens(system, kept) >= MIN_FILL * budget else None


def build_story(story_dir: Path) -> dict[Path, str]:
    """{output path: text} for one story."""
    card, messages = load_story(story_dir)
    system = card["system"].strip()
    probes_text = (story_dir / "probes.yaml").read_text(encoding="utf-8")
    probes = yaml.safe_load(probes_text.replace("__NPCS__", "|".join(card["npc_names"])))
    defaults = probes["defaults"]
    files: dict[Path, str] = {}
    conditions = probes.get("conditions") or []
    for cond in conditions:
        if any(rx.search(cond["note"]) for rx in map(re.compile, card["name_patterns"])):
            raise BuildError(f"condition {cond['key']}: the note gives the name away")
    for length, (budget, timeout) in BUCKETS.items():
        history = cut(system, messages, budget)
        if history is None:
            continue
        fixture = {"name": card["name"], "length": length,
                   "approx_tokens": _tokens(system, history),
                   "system": system, "messages": history}
        rel = f"fixtures/{card['name']}.{length}.json"
        files[OUT / rel] = json.dumps(fixture, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
        variants = [(probe, None) for probe in probes["probes"]]
        variants += [(probe, cond) for cond in conditions if length in cond["lengths"]
                     for probe in probes["probes"] if probe["key"] in cond["probes"]]
        for probe, cond in variants:
            name = probe["key"] + (f"-{cond['key']}" if cond else "")
            sid = f"rp-{name}-{length}"
            follow_ups = probe.get("follow_ups") or []
            generations = 1 + len(follow_ups)
            sampling = dict(defaults["sampling"])
            if probe.get("max_tokens"):
                sampling["max_tokens"] = probe["max_tokens"]
            scenario = {
                "id": sid,
                "title": f"{probe['title']} ({length} context"
                         + (f", with {cond['key']})" if cond else ")"),
                "tier": defaults["tier"], "category": "rp_long_context", "domain": "generic",
                "description": probe["description"],
                "tags": probe["tags"] + ([f"condition_{cond['key']}"] if cond else []),
                # Unranked: read these as rates, with `toolery rp-report`.
                "ranking_dimensions": [],
                "prompt": (probe["prompt"].rstrip() + "\n\n" + cond["note"] + "\n") if cond
                          else probe["prompt"],
                "tools": [],
                # One request per reply; later replies reuse the server's
                # prompt cache, so the time allowed grows by half a turn each.
                "budget": {"max_tool_calls": 0, "max_turns": generations,
                           "timeout_seconds": timeout * (1 + generations) // 2},
                "tool_responses": {},
                "conversation": {"fixture": rel},
                "sampling": sampling,
                "scoring": probe["scoring"],
            }
            if follow_ups:
                scenario["follow_ups"] = follow_ups
            header = (f"# GENERATED by scripts/build_rp_fixtures.py from "
                      f"fixtures/src/{story_dir.name}/ — do not edit by hand.\n")
            files[OUT / f"{sid}.yaml"] = header + yaml.dump(
                scenario, Dumper=_NoAlias, sort_keys=False, allow_unicode=True, width=10_000)
    return files


class _NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _block_str(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_NoAlias.add_representer(str, _block_str)


def build_all() -> dict[Path, str]:
    files: dict[Path, str] = {}
    for story_dir in sorted(p for p in SRC.iterdir() if (p / "card.yaml").exists()):
        files.update(build_story(story_dir))
    return files


def stale(files: dict[Path, str]) -> list[Path]:
    """Outputs that are missing, differ, or should no longer exist. Line
    endings are ignored: a checkout may rewrite them."""
    def norm(text: str) -> str:
        return text.replace("\r\n", "\n")
    bad = [p for p, text in files.items()
           if not p.exists() or norm(p.read_text(encoding="utf-8")) != text]
    if OUT.exists():
        bad += [p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".yaml", ".json")
                and p not in files]
    return sorted(bad)


def main(argv: list[str]) -> int:
    files = build_all()
    out_of_date = stale(files)
    if "--check" in argv:
        for p in out_of_date:
            print(f"out of date: {p.relative_to(ROOT)}")
        return 1 if out_of_date else 0
    for p in out_of_date:
        if p in files:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(files[p], encoding="utf-8", newline="\n")
        else:
            p.unlink()
    for p, text in sorted(files.items()):
        if p.suffix == ".json":
            f = json.loads(text)
            print(f"{p.relative_to(ROOT)}: {len(f['messages']) // 2} exchanges, "
                  f"~{f['approx_tokens']} tokens")
    print(f"{len(files)} files, {len(out_of_date)} written or removed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
