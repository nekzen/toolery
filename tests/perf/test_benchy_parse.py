"""parse_benchy_json against a fixture built with llama-benchy 0.4.0's own
pydantic models (BenchmarkReport/BenchmarkRun), so the test tracks the real
schema: time to first token is `e2e_ttft` (there is no `ttft` field), metrics
are {mean, std, values}, and depths > 0 may carry a context-prefill run."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from toolery.perf.benchy import BenchyResult, parse_benchy_json

FIXTURE = Path(__file__).parent / "fixtures" / "llama_benchy_0.4.0.json"


def _rows():
    return parse_benchy_json(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_one_row_per_depth_context_prefill_run_skipped():
    assert [r["depth"] for r in _rows()] == [0, 4096, 131072]


def test_ttft_comes_from_e2e_ttft():
    d0 = _rows()[0]
    assert d0["ttft_ms"] == 200.0
    assert d0["ttft_p95_ms"] == 220.0
    assert d0["pp_tps"] == 2500.0 and d0["tg_tps"] == 41.0 and d0["n_runs"] == 3


def test_depth_4096_reports_the_real_test_not_the_prefill_phase():
    d = _rows()[1]
    assert d["pp_tps"] == 2250.0          # the test, not the ctx_pp 2350.0
    assert d["tg_tps"] == 39.0            # the prefill phase generates nothing
    assert d["ttft_ms"] == 2000.0


def test_depth_without_data_keeps_none_not_zero():
    d = _rows()[2]
    assert d["pp_tps"] is None and d["tg_tps"] is None
    assert d["ttft_ms"] is None and d["ttft_p95_ms"] is None


def test_perf_command_prints_na_for_missing_metrics(monkeypatch):
    """Regression: `toolery perf` crashed with "unsupported format string
    passed to NoneType.__format__" as soon as a metric was None."""
    import toolery.perf.benchy as benchy
    from toolery.cli import app

    monkeypatch.setattr(benchy, "run_benchy",
                        lambda **kw: BenchyResult(model="m", rows=_rows()))
    out = CliRunner().invoke(app, ["perf", "--model", "m"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert "ttft=200ms" in out.output
    assert "depth= 131072 pp_tps=n/a tg_tps=n/a ttft=n/a" in out.output
