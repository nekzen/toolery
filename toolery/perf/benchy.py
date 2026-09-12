from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _mean(metric: dict | None) -> float | None:
    return metric.get("mean") if isinstance(metric, dict) else None


def _p95(metric: dict | None) -> float | None:
    """95th percentile (nearest rank) of a llama-benchy BenchmarkMetric's
    per-request ``values`` — the metric itself only carries mean/std."""
    values = sorted((metric or {}).get("values") or []) if isinstance(metric, dict) else []
    if not values:
        return None
    return values[max(0, math.ceil(0.95 * len(values)) - 1)]


def parse_benchy_json(data: dict) -> list[dict]:
    """Flatten a llama-benchy JSON report into one row per measured depth
    (pp_tps, tg_tps, ttft_ms, ...).

    llama-benchy's BenchmarkRun (0.3.x-0.4.x) reports time to first token as
    ``e2e_ttft`` (ms) — there is no ``ttft`` field — and each metric is
    {mean, std, values}, with no precomputed p95. For depths > 0 it can also
    emit a separate context-prefill run (``is_context_prefill_phase``,
    "ctx_pp @ dN") before the real test at the same depth: that run only
    measures loading the context and generates nothing, so it is skipped.
    A metric is None when a depth produced no data (e.g. it did not fit in
    the server's context); the row keeps None rather than a fake 0.
    """
    raw_rows = data.get("benchmarks") or data.get("runs") or []
    rows = []
    for r in raw_rows:
        if "pp_throughput" not in r:  # legacy schema — pass through
            rows.append(r)
            continue
        if r.get("is_context_prefill_phase"):
            continue
        ttft = r.get("e2e_ttft") or r.get("ttft")
        rows.append({
            "depth": r.get("context_size", 0),
            "pp_tps": _mean(r.get("pp_throughput")),
            "tg_tps": _mean(r.get("tg_throughput")),
            "ttft_ms": _mean(ttft),
            "ttft_p95_ms": _p95(ttft),
            "pp_tokens": r.get("prompt_size"),
            "tg_tokens": r.get("response_size"),
            "n_runs": len((r.get("pp_throughput") or {}).get("values") or []),
        })
    return rows


@dataclass
class BenchyResult:
    model: str
    rows: list[dict]


def run_benchy(*, model: str, base_url: str, pp: int = 4096, tg: int = 512,
               depth: list[int] | None = None, runs: int = 3,
               output_file: Path | None = None,
               extra_args: list[str] | None = None) -> BenchyResult:
    depth = depth or [0, 4096, 8192]
    if output_file is None:
        output_file = Path(tempfile.mkstemp(suffix=".json")[1])
    # llama-benchy >=0.3.8 hits {base_url}/chat/completions directly (no /v1
    # prefix added). vLLM exposes those endpoints under /v1, so make sure the
    # base_url passed downstream ends with /v1.
    benchy_base = base_url.rstrip("/")
    if not benchy_base.endswith("/v1"):
        benchy_base += "/v1"
    # Prefer the llama-benchy installed in the current environment (the
    # version pinned by `uv sync --extra perf`); `uvx` is only the fallback —
    # it fetches the LATEST PyPI release into an isolated env, so its output
    # schema can drift ahead of what the parser below understands.
    benchy_exe = shutil.which("llama-benchy")
    launcher = [benchy_exe] if benchy_exe else ["uvx", "llama-benchy"]
    cmd = [
        *launcher,
        "--base-url", benchy_base, "--model", model,
        "--pp", str(pp), "--tg", str(tg),
        "--depth", *(str(d) for d in depth),
        "--runs", str(runs),
        "--format", "json", "--save-result", str(output_file),
    ]
    if extra_args:
        cmd += extra_args
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        # llama-benchy prints transformers warnings to stderr first and the real
        # failure (e.g. a connection error) to stdout last — so report the TAIL
        # of the combined output, not the head of stderr, which is just noise.
        combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        tail = "\n".join(combined.splitlines()[-8:]) or "(no output)"
        hint = ""
        if "connect" in combined.lower():
            hint = (f"\n\nHint: could not reach the model server at {benchy_base}. "
                    "Make sure it is running and that --base-url / TOOLERY_BASE_URL is correct.")
        raise RuntimeError(
            f"llama-benchy failed (exit {completed.returncode}):\n{tail}{hint}"
        )
    data = json.loads(Path(output_file).read_text(encoding="utf-8"))
    rows = parse_benchy_json(data)
    return BenchyResult(model=data.get("model", model), rows=rows)
