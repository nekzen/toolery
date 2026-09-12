"""Concurrency guards for the shared runs.db — several toolery processes
(parallel runs, the TUI poller, per-run heartbeat threads) write to it at
once. Separate Store instances open separate connections, which contend for
the database lock exactly like separate processes do."""
from __future__ import annotations

import threading
from datetime import UTC, datetime

from toolery.cli import _claim_run_id
from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import BUSY_TIMEOUT_S, Store


def _result(sid: str, trial: int) -> ScenarioResult:
    trace = TraceResult(scenario_id=sid, adapter="raw", trial_index=trial,
                        messages=[Message(role="user", content="hi")], tool_calls=[],
                        final_response="ok", started_at_iso="2026-09-12T00:00:00Z",
                        duration_ms=10, error=None)
    return ScenarioResult(scenario_id=sid, adapter="raw", trial_index=trial,
                          status="pass", score=1.0, call_count=1, budget_max=1,
                          latency_ms=10, failure_kind=None, checks=[], trace=trace)


def test_init_schema_enables_wal(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    with store.conn() as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == int(BUSY_TIMEOUT_S * 1000)


def test_parallel_runs_write_without_locking_errors(tmp_path):
    db = tmp_path / "runs.db"
    Store(db).init_schema()
    n_runs, n_results = 4, 60
    errors: list[BaseException] = []

    def one_run(k: int) -> None:
        # Each "run" gets its own Store (own connections), plus a heartbeat
        # thread hammering runs.updated_at — the real multi-process pattern.
        store = Store(db)
        run_id = f"run-{k}"
        try:
            store.create_run(run_id=run_id, model=f"m{k}", base_url="x",
                             started_at=datetime.now(UTC).isoformat(),
                             config_json="{}", scenarios_hash="h")
            stop = threading.Event()

            def heartbeat() -> None:
                while not stop.is_set():
                    store.heartbeat(run_id)

            hb = threading.Thread(target=heartbeat)
            hb.start()
            for i in range(n_results):
                sid = f"s-{i:03d}"
                store.mark_in_flight(run_id, sid, "raw", 0, datetime.now(UTC).isoformat())
                store.write_scenario_result(run_id, _result(sid, 0), tags=[],
                                            ranking_dims=["overall"], scenario_hash="h",
                                            category="coding", tier="easy",
                                            trace_path="t.json")
                store.clear_in_flight(run_id, sid, "raw", 0)
            stop.set()
            hb.join()
            store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)
        except BaseException as e:  # surfaced to the main thread below
            errors.append(e)

    def reader() -> None:
        # The TUI polls while runs write; WAL means it never blocks them.
        store = Store(db)
        try:
            for _ in range(200):
                store.fetch_all_runs()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=one_run, args=(k,)) for k in range(n_runs)]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    store = Store(db)
    for k in range(n_runs):
        assert store.count_results_for_run(f"run-{k}") == n_results


def test_claim_run_id_is_unique_within_the_same_second(tmp_path):
    runs = tmp_path / "runs"
    now = datetime(2026, 9, 12, 14, 30, 5, tzinfo=UTC)
    first = _claim_run_id(runs, "Qwen3.6-27B", now)
    second = _claim_run_id(runs, "Qwen3.6-27B", now)
    third = _claim_run_id(runs, "Qwen3.6-27B", now)
    assert first == "2026-09-12T14-30-05_Qwen3.6-27B"
    assert second == "2026-09-12T14-30-05_Qwen3.6-27B-2"
    assert third == "2026-09-12T14-30-05_Qwen3.6-27B-3"
    assert all((runs / r).is_dir() for r in (first, second, third))


def test_claim_run_id_is_unique_across_concurrent_claimers(tmp_path):
    runs = tmp_path / "runs"
    now = datetime(2026, 9, 12, 14, 30, 5, tzinfo=UTC)
    claimed: list[str] = []
    lock = threading.Lock()

    def claim() -> None:
        rid = _claim_run_id(runs, "same-model", now)
        with lock:
            claimed.append(rid)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == len(set(claimed)) == 8


def test_open_reader_does_not_block_a_writer(tmp_path):
    """The real-world lock: the TUI scans scenario_results every few seconds.
    In rollback-journal mode a reader's SHARED lock stops a writer from
    committing, and after sqlite3's 5s default the run crashes with
    "database is locked". In WAL mode the writer commits immediately."""
    import sqlite3
    import time

    db = tmp_path / "runs.db"
    store = Store(db)
    store.init_schema()
    store.create_run(run_id="r", model="m", base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")

    reader = sqlite3.connect(db)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM scenario_results").fetchone()
    try:
        t0 = time.monotonic()
        store.write_scenario_result("r", _result("s-000", 0), tags=[],
                                    ranking_dims=["overall"], scenario_hash="h",
                                    category="coding", tier="easy", trace_path="t.json")
        assert time.monotonic() - t0 < 2.0
    finally:
        reader.rollback()
        reader.close()
    assert store.count_results_for_run("r") == 1
