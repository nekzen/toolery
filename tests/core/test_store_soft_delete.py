"""Soft-delete: deleted_at column, delete_run()/restore_run(), fetch_all_runs()."""
from toolery.core.store import Store


def _store(tmp_path) -> Store:
    s = Store(tmp_path / "runs.db")
    s.init_schema()
    return s


def test_deleted_at_column_exists_and_defaults_null(tmp_path):
    s = _store(tmp_path)
    s.create_run(run_id="r1", model="m", base_url="u",
                 started_at="2026-01-01T00:00:00Z", config_json="{}", scenarios_hash="h")
    with s.conn() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
    assert "deleted_at" in cols
    run = s.fetch_run("r1")
    assert run["deleted_at"] is None


def test_delete_run_hides_from_fetch_all_runs_by_default(tmp_path):
    s = _store(tmp_path)
    s.create_run(run_id="r1", model="m", base_url="u",
                 started_at="2026-01-01T00:00:00Z", config_json="{}", scenarios_hash="h")
    s.create_run(run_id="r2", model="m2", base_url="u",
                 started_at="2026-01-02T00:00:00Z", config_json="{}", scenarios_hash="h")
    s.delete_run("r1")

    active = s.fetch_all_runs()
    assert {r["run_id"] for r in active} == {"r2"}

    with_deleted = s.fetch_all_runs(include_deleted=True)
    assert {r["run_id"] for r in with_deleted} == {"r1", "r2"}
    deleted_row = next(r for r in with_deleted if r["run_id"] == "r1")
    assert deleted_row["deleted_at"] is not None


def test_restore_run_undoes_soft_delete(tmp_path):
    s = _store(tmp_path)
    s.create_run(run_id="r1", model="m", base_url="u",
                 started_at="2026-01-01T00:00:00Z", config_json="{}", scenarios_hash="h")
    s.delete_run("r1")
    assert s.fetch_all_runs() == []

    s.restore_run("r1")
    active = s.fetch_all_runs()
    assert {r["run_id"] for r in active} == {"r1"}
    assert s.fetch_run("r1")["deleted_at"] is None


def test_old_db_migrates_deleted_at_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY, model TEXT NOT NULL, base_url TEXT,
          started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP, duration_s REAL,
          status TEXT, config_json TEXT, llm_test_version TEXT, scenarios_hash TEXT
        );
    """)
    con.commit()
    con.close()

    store = Store(db)
    store.init_schema()  # must not raise
    with store.conn() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
    assert "deleted_at" in cols
