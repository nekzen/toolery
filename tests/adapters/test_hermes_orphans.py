"""The hermes adapter must not leave Hermes — or the MCP server Hermes spawns —
running after a trial ends, whatever the way it ends. Orphans kept using the
model server and slowed every following trial.

A fake Hermes stands in for the real CLI: it spawns a child (like Hermes'
MCP server) and both append a heartbeat to a file while alive. If a file
still grows after the trial ended, a process survived. Both scripts exit on
their own after a while, so a failing run leaves no permanent orphans.
"""
from __future__ import annotations

import asyncio
import sys
import textwrap
import time
from pathlib import Path

import pytest

from toolery.adapters.hermes import HermesAdapter
from toolery.core.models import Budget, Category, Scenario, Scoring, Tier

LIFETIME_S = 20  # fake processes give up on their own after this

_BEAT = textwrap.dedent("""
    import sys, time
    path, lifetime = sys.argv[1], float(sys.argv[2])
    end = time.time() + lifetime
    while time.time() < end:
        with open(path, "a") as f:
            f.write("x")
        time.sleep(0.05)
""")

_FAKE_HERMES = textwrap.dedent("""
    import subprocess, sys, time
    beat, workdir, lifetime, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    # The "MCP server": a child of Hermes, not of toolery. Like a stdio MCP
    # server it has its own pipes, so it does not hold Hermes' stdout open.
    subprocess.Popen([sys.executable, beat, workdir + "/child.hb", str(lifetime)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    if mode == "exit":
        time.sleep(0.5)
        print("done")
        sys.exit(0)
    end = time.time() + lifetime
    while time.time() < end:
        with open(workdir + "/hermes.hb", "a") as f:
            f.write("x")
        time.sleep(0.05)
""")


def _scenario() -> Scenario:
    return Scenario(
        id="t-01-hermes", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p", tools=[],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        tool_responses={}, scoring=Scoring())


def _cmd(tmp_path: Path, mode: str) -> list[str]:
    (tmp_path / "beat.py").write_text(_BEAT, encoding="utf-8")
    (tmp_path / "fake_hermes.py").write_text(_FAKE_HERMES, encoding="utf-8")
    return [sys.executable, str(tmp_path / "fake_hermes.py"), str(tmp_path / "beat.py"),
            str(tmp_path), str(LIFETIME_S), mode]


def _sizes(tmp_path: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in tmp_path.glob("*.hb")}


async def _assert_all_dead(tmp_path: Path) -> None:
    await asyncio.sleep(0.6)                 # let any survivor write again
    before = _sizes(tmp_path)
    assert "child.hb" in before, "the fake MCP server never started"
    await asyncio.sleep(0.8)
    after = _sizes(tmp_path)
    survivors = [name for name in after if after[name] != before.get(name)]
    assert not survivors, f"still running after the trial ended: {survivors}"


async def _wait_for_child(tmp_path: Path) -> None:
    deadline = time.monotonic() + 10
    while "child.hb" not in _sizes(tmp_path) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_runner_cancellation_kills_hermes_and_its_children(tmp_path):
    """The runner stops a trial at the scenario budget by cancelling it."""
    adapter = HermesAdapter(cli_path=sys.executable, timeout_per_scenario=1800)
    task = asyncio.create_task(adapter._execute_and_trace(
        _scenario(), "m", _cmd(tmp_path, "hang"), time.monotonic(), env=None, use_lock=False))
    await _wait_for_child(tmp_path)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_all_dead(tmp_path)


@pytest.mark.asyncio
async def test_own_timeout_kills_the_whole_tree(tmp_path):
    adapter = HermesAdapter(cli_path=sys.executable, timeout_per_scenario=2)
    trace = await adapter._execute_and_trace(
        _scenario(), "m", _cmd(tmp_path, "hang"), time.monotonic(), env=None, use_lock=False)
    assert trace.error == "hermes: timeout"
    await _assert_all_dead(tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason=(
    "once the parent has exited, Windows can no longer find its children by "
    "PID; the straggler sweep relies on POSIX process groups (Hermes runs on "
    "Linux, where Hermes itself also shuts its MCP server down on exit)"))
@pytest.mark.asyncio
async def test_normal_exit_leaves_no_stragglers(tmp_path):
    adapter = HermesAdapter(cli_path=sys.executable, timeout_per_scenario=1800)
    trace = await adapter._execute_and_trace(
        _scenario(), "m", _cmd(tmp_path, "exit"), time.monotonic(), env=None, use_lock=False)
    assert trace.error is None
    await _assert_all_dead(tmp_path)


def test_never_signals_a_group_it_cannot_confirm():
    """os.killpg on a mocked or bogus pid can hit toolery's own process group:
    an AsyncMock's pid converts to 1, and killpg(0) means "my own group". The
    group is only used when it is a real child-led group, never our own."""
    import os
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from toolery.adapters.hermes import _child_group

    assert _child_group(AsyncMock()) is None                       # mocked pid
    assert _child_group(SimpleNamespace(pid=0)) is None            # "my own group"
    assert _child_group(SimpleNamespace(pid=1)) is None
    assert _child_group(SimpleNamespace(pid=os.getpid())) is None  # our own group
