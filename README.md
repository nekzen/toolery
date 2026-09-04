# Toolery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**Toolery is a deterministic LLM tool-calling benchmark.** It runs a model
through a fixed, versioned set of scenarios that each expose a small set of
mock tools, scores every trial with deterministic assertions (no LLM judge —
$0 cost, fully reproducible), and aggregates results into per-model rankings
across a matrix of capability dimensions.

---

## Overview

Toolery ships **200 scenarios** across **4 difficulty tiers**
(`easy` / `medium` / `hard` / `very_hard`) and **20 categories** (coding,
debugging, security review, data analysis, fact verification, and more).
Each scenario is a self-contained YAML file describing:

- a **prompt** the model receives,
- the **tools** it may call (mock implementations — no network/DB access),
- **canned tool responses** the mock runtime returns when the model calls
  those tools,
- a **budget** (max tool calls, max turns, timeout),
- and a set of **scoring checks** (required / forbidden / partial) that
  deterministically grade the transcript.

Toolery talks to the model under test through an **adapter**:

| Adapter | What it does |
|---|---|
| `raw` | Calls a local OpenAI-compatible endpoint directly with the standard `tools` API. The baseline measurement of a model's own tool-calling. |
| `cloud` | Same OpenAI-compatible protocol, against a remote/hosted API (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). |
| `hermes` | Spawns the `hermes` CLI as a subprocess and reconstructs the trace from its session store — measures what an agent harness adds or breaks relative to `raw`. |

Every trial is scored by the checks declared in the scenario's YAML
(see [Check types](#check-types)), rolled up into a pass/partial/fail
status, and aggregated into **ranking dimensions** — coding, debugging,
safety, tool selection, etc. — that make up the capability matrix (see
[Ranking dimensions](#ranking-dimensions)). On top of the raw scores,
Toolery can also evaluate a run against **role-based thresholds** (Coder,
Security-Auditor, Data-Analyst, …) to answer "is this model adequate for
job X?" (see [Role-based thresholds](#role-based-thresholds)).

---

## Installation

Requires **Python 3.11+**. The project uses
[uv](https://docs.astral.sh/uv/) and a `hatchling` build backend.

```bash
# From PyPI (if published)
pip install toolery

# With uv (recommended — manages an isolated virtualenv for you)
uv pip install toolery

# From source
git clone https://github.com/nekzen/toolery.git
cd toolery
uv sync                 # base install
uv sync --extra dev     # + pytest, ruff, mypy (needed for tests)
uv sync --extra perf    # + llama-benchy (throughput benchmarking)
```

Everything below is invoked as `uv run toolery …` when installed from
source (no manual venv activation needed). If you `pip install`ed into an
active environment, drop the `uv run` prefix and call `toolery` directly.

---

## Quick start

```bash
# 1. Point Toolery at your running model server
export TOOLERY_BASE_URL=http://localhost:8000

# 2. Smoke test: a handful of easy scenarios, raw adapter, 3 trials each
uv run toolery run --model my-model --adapter raw --tier easy --trials 3

# 3. Full run: all tiers, 5 trials
uv run toolery run --model my-model --adapter raw --tier all --trials 5

# 4. List recorded runs
uv run toolery list

# 5. Regenerate and view the ranking tables
uv run toolery rankings --regen
uv run toolery rankings --dimension overall

# 6. Or explore everything in the terminal UI
uv run toolery tui
```

A run writes scenario traces and scores under `./results/` (SQLite +
per-run JSON) and regenerates ranking markdown under `results/rankings/`.

---

## CLI commands

All commands are subcommands of `toolery` (`uv run toolery <command>
--help` for the authoritative, live option list).

### `toolery list`

List recorded benchmark runs.

```bash
toolery list [--json] [--include-deleted]
```

| Flag | Meaning |
|---|---|
| `--json` | Emit machine-readable JSON instead of a table. |
| `--include-deleted` | Also show soft-deleted runs. |

### `toolery scenarios`

List scenarios available in the scenario set on disk.

```bash
toolery scenarios [--tier easy|medium|hard|very_hard|all] [--json]
```

| Flag | Meaning |
|---|---|
| `--tier` | Filter by difficulty tier (default `all`). |
| `--json` | Emit machine-readable JSON (`id`, `tier`, `category`, `domain`, `tools`, `title`) instead of a table. |
| `dir` (positional-less option, defaults to `scenarios`) | Override the scenarios directory. |

### `toolery run`

Run scenarios against a model and score them.

```bash
toolery run --model my-model [options]
```

| Flag | Meaning |
|---|---|
| `--model` | Friendly display name, used as the run's identity in the DB. Required unless `--resume` is given. |
| `--served-model` | Model name sent as `model=` in the API request (defaults to `--model`). Use when the display name and the served model id differ. |
| `--adapter` | Comma-separated: `raw`, `cloud`, `hermes` (default `raw`). Multiple adapters run the same scenario set once per adapter. |
| `--tier` | `easy \| medium \| hard \| very_hard \| all` (default `all`). |
| `--category` | Scenario category filter (see [Categories](#categories)) or `all`. |
| `--ids` | Comma-separated exact scenario ids to run; empty means no id filtering. |
| `--trials` | Number of trials per scenario (default 5). |
| `--concurrency` | Number of scenarios executed in parallel (default 4). |
| `--timeout-scale` | Multiplier applied to each scenario's `timeout_seconds` (default 2.0). Raise for slow cloud/reasoning endpoints. |
| `--base-url` | Endpoint for `raw`/`cloud` adapters (default `http://localhost:8000`). |
| `--cluster` | Deployment topology label: `single \| dual \| triple \| quad \| octa`. Purely metadata — tracks which node configuration produced a run. |
| `--json` | With `--dry-run`, emit the plan as JSON instead of text. |
| `--dry-run` | Validate the scenario/adapter/filter selection and print the planned unit count without executing anything. |
| `--max-retries` | Retry attempts for **transient** adapter failures only (HTTP 429, timeout, connection reset, 5xx). Genuine model failures (bad tool call, wrong answer) are never retried. Default 0 (no retries). |
| `--retry-backoff-base` | Seconds — base of the exponential backoff between retries (`base * 2**attempt`). Default 1.0. |
| `--retry-backoff-max` | Seconds — cap on the exponential backoff delay. Default 30.0. |
| `--resume <run_id>` | Rehydrate model/adapter/tier/trials/etc. from a previous run's stored config and continue from the next not-yet-run unit. |
| `--with-perf` | Also run the llama-benchy throughput benchmark for this run. |
| `--perf-only` | Skip the eval phase; run only llama-benchy. |

Not every flag listed above appears literally in `--help` (some, like
`--with-tools`, do not exist as scenario filters on `run`/`scenarios`
today — filter by `--ids` or `--category` instead).

### `toolery compare`

Diff two runs, including McNemar significance.

```bash
toolery compare <run_a> <run_b> [--out path.md] [--json]
```

| Flag | Meaning |
|---|---|
| `--out` | Output path for the markdown report (default `results/compare/<a>__vs__<b>.md`). |
| `--json` | Emit the comparison as JSON to stdout instead of writing a markdown report. |

### `toolery rankings`

Regenerate or inspect the ranking tables.

```bash
toolery rankings [--regen] [--dimension overall|coding|...|all]
```

| Flag | Meaning |
|---|---|
| `--regen` | Recompute rankings from stored runs and write markdown under `results/rankings/`. |
| `--dimension` | Restrict to one dimension, or `all` (default) for every dimension in [Ranking dimensions](#ranking-dimensions). |

### `toolery tui`

```bash
toolery tui
```

Launches the Textual terminal dashboard: discover endpoints, launch runs,
and browse rankings/scenarios/history without leaving the terminal.

### `toolery perf`

Run the llama-benchy throughput benchmark only (no scoring).

```bash
toolery perf --model my-model [--base-url http://localhost:8000] [--pp 4096] [--tg 512] [--depth 0,4096,8192] [--runs 3]
```

### `toolery roles list`

List all available role profiles and their required thresholds.

```bash
toolery roles list
```

### `toolery roles check`

Check whether a run meets a role's minimum pass-rate thresholds.

```bash
toolery roles check <run_id> <role_key> [--json]
```

Prints a per-category table (required threshold vs. actual pass rate) and
an overall verdict: **ADEQUATE** or **NOT ADEQUATE**. Exits non-zero when
the verdict is NOT ADEQUATE (useful in CI-style gating). See
[Role-based thresholds](#role-based-thresholds).

### `toolery roles rank`

Rank all (model, adapter) pairs by a role's weighted score.

```bash
toolery roles rank <role_key> [--regen] [--top 20]
```

### Run lifecycle: delete / restore

```bash
toolery delete-run <run_id>     # soft-delete: hidden from `list` unless --include-deleted
toolery restore-run <run_id>    # undo a soft-delete
```

Soft-delete is reversible; there is no separate hard-delete/purge command —
to permanently remove a run's data, delete its rows/files directly from the
`results/` store.

---

## Scenario format

Scenarios live under `scenarios/<category>/*.yaml`, one file per scenario.
**All scenario prompts, titles, and descriptions must be written in
English** — this keeps checks (regex, keyword, structured-output matching)
consistent across the whole set. Non-English *variants* of a scenario are
supported explicitly via the `language` field (see below), not by writing
the base scenario in another language.

### YAML structure

```yaml
# scenarios/code_review/code-review-easy-01-sql-injection-python.yaml
id: code-review-easy-01-sql-injection-python   # unique, matches the filename (no .yaml)
title: "Spot a SQL injection vulnerability in a Python snippet"
tier: easy                                     # easy | medium | hard | very_hard
category: code_review                          # must match the containing directory name
domain: dev_ops                                # free-text grouping tag (e.g. dev_ops, quant, travel)
# language: fr                                 # optional — see "Multilingual variants" below

description: |
  One or two sentences explaining what the scenario is testing and why.

tags: [code_review, security, sql_injection]   # free-text tags, informational only

ranking_dimensions: [overall, coding, safety]  # legacy dimension tags; category-derived
                                                # dimensions (e.g. "code_review") are inferred
                                                # automatically from `category` — see below

prompt: |
  The exact text sent to the model. Must be in English. Include any code
  snippets, data, or context the model needs inline.

tools: [read_file, grep]                       # subset of the registered mock tools the
                                                # model is allowed to see/call (empty list [] if none)

budget:
  max_tool_calls: 2
  max_turns: 3
  timeout_seconds: 45

tool_responses:                                # canned responses the mock runtime returns
  read_file:
    - match: { path: "app.py" }                # match on specific call arguments
      returns: { content: "..." }
    - match: any                                # fallback for any unmatched call
      returns: { content: "" }

scoring:
  required:                                    # ALL must pass for the trial to pass
    - check: response_satisfies
      any_of: [["SQL injection", "sql injection"]]
  forbidden: []                                # ANY passing means the trial fails
  partial:                                     # optional: partial credit if required checks
    - check: response_satisfies                # fail but these pass
      any_of: [["f-string", "string interpolation"]]
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

### Tiers

| Tier | Meaning |
|---|---|
| `easy` | Single obvious tool call or a direct answer; minimal ambiguity. |
| `medium` | Multiple tool calls or one non-trivial parameter/format constraint. |
| `hard` | Multi-step chains, error handling, or tight budgets. |
| `very_hard` | Long-horizon planning, adversarial input, or several compounding constraints. |

### Categories

All 20 categories (directory name under `scenarios/` == `category` value):

| Category | Focus |
|---|---|
| `tool_selection` | Picking the right tool among plausible distractors. |
| `parameter_precision` | Exact argument values — units, codes, formats, numeric bounds. |
| `multi_step_chains` | Sequential/parallel multi-tool workflows. |
| `restraint_refusal` | Knowing when *not* to call a tool. |
| `error_recovery` | Recovering from timeouts, 429s, malformed/partial tool results. |
| `structured_reasoning` | Step-by-step reasoning before acting. |
| `instruction_following` | Strict format/length/negative-constraint compliance. |
| `context_state_tracking` | Reusing prior turns'/tool calls' results correctly. |
| `coding` | TDD loops, refactors, file operations, git discipline. |
| `debugging` | Root-cause analysis from tracebacks/regressions. |
| `safety_boundaries` | Refusing an explicit unsafe request. |
| `adversarial_robustness` | Resisting prompt injection in untrusted tool/RAG output. |
| `toolset_scale` | Behaving correctly with large tool catalogs. |
| `autonomous_planning` | Open-ended multi-step planning and composition. |
| `creative_composition` | Generating styled/creative content on request. |
| `structured_output` | Non-JSON structured formats — CSV, YAML, markdown tables. |
| `hallucination` | Calibration — refusing/hedging on ungrounded questions instead of fabricating. |
| `terminal_handling` | Shell pipes, CLI/ANSI parsing, destructive-command refusal. |
| `fact_verification` | Verifying claims against provided evidence. |
| `creative_writing` | Long-form styled prose generation. |
| `code_review` | Spotting bugs/vulnerabilities in given code. |
| `workflow_orchestration` | Coordinating multi-tool business workflows. |
| `security_audit` | Security-focused review of code/configs/infrastructure. |
| `data_analysis` | Extracting/summarizing insight from structured data. |

### Available mock tools

Registered under `toolery/tools/` (`generic.py`, `domain.py`, `terminal.py`,
`api_db.py`). List a subset of these under a scenario's `tools:` key.

**Generic / everyday** (`generic.py`): `get_weather`, `web_search`,
`send_email`, `get_contacts`, `calculator`, `read_file`, `write_file`,
`list_files`, `add_calendar_event`, `get_exchange_rate`, `get_stock_price`.

**Dev / ops / finance / travel** (`domain.py`): `git_status`, `git_diff`,
`git_add`, `git_commit`, `git_branch`, `git_log`, `git_show`, `grep`,
`edit_file`, `run_tests`, `run_lint`, `run_bash`, `python_exec`,
`get_order_status`, `get_account`, `get_positions`, `get_orderbook`,
`get_risk`, `submit_order`, `transfer_funds`, `search_flights`,
`get_weather_global`, `deploy`, `delete_user`, `admin_grant_role`,
`vllm_config_get`, `vllm_config_set`.

**Terminal** (`terminal.py`): `bash_exec`, `process_start`,
`process_status`, `process_kill`, `process_send_input`,
`read_tty_buffer`.

**HTTP / SQL mocks** (`api_db.py`): `http_get`, `http_post`,
`http_paginate`, `sql_query`, `sql_describe`, `db_list_tables`.

All tools are pure mocks with no real network/filesystem/DB access —
their return values are driven entirely by each scenario's
`tool_responses` block.

### Check types

Set in `scoring.required` / `scoring.forbidden` / `scoring.partial`. Each
entry is `{check: <name>, ...check-specific fields}`.

| Check | Purpose |
|---|---|
| `tool_called` | The named tool was called at least once. |
| `tool_not_called` | The named tool was never called. |
| `tool_args_contain` | A tool call's arguments contain given key/value pairs. |
| `call_count_at_most` / `call_count_at_least` / `call_count_exactly` | Bound the number of calls to a tool. |
| `tool_called_in_order` | A sequence of tools was called in the given order. |
| `tool_called_in_parallel` | A set of tools was called without sequential dependency. |
| `tool_args_match_regex` | A tool call argument matches a regex. |
| `tool_args_type` | A tool call argument has the expected type. |
| `command_regex_match` | A shell/terminal command argument matches a regex (with destructive-command awareness). |
| `ansi_stripped_in_response` | The final response has ANSI escape sequences stripped from any echoed terminal output. |
| `no_destructive_command` | No destructive shell command (`rm -rf`, etc.) was issued without confirmation. |
| `response_contains` / `response_not_contains` | Substring presence/absence in the final response. |
| `response_matches_schema` | Final response (JSON) validates against a JSON Schema. |
| `response_number` | Final response contains a number equal to (± tolerance) an expected value. |
| `response_csv` | Final response parses as CSV with expected shape. |
| `response_yaml` | Final response parses as valid YAML. |
| `response_markdown_table` | Final response contains a markdown table with expected columns. |
| `response_language` | Final response is written in the expected language (`language: <code>`). |
| `unique_tools_called` | Only distinct tools were called (no redundant repeats). |
| `no_hallucinated_tool` | No tool call references a tool name that wasn't offered. |
| `budget_respected` | Tool-call/turn budget was not exceeded. |
| `clarification_asked` | The model asked a clarifying question when required. |
| `error_surfaced` | A tool error was surfaced to the user rather than silently swallowed. |
| `final_state_equals` | Tracked mock state (e.g. a mock DB row) ended in the expected value. |
| `response_satisfies` | Keyword/phrase matching over `all_of` / `any_of` / `none_of` groups. |
| `response_matches_regex` | Same `all_of`/`any_of`/`none_of` shape as `response_satisfies`, but entries are regexes. |
| `response_diff` | **(new)** Flags near-duplicate responses against a `reference` string (Jaccard token-overlap or cosine bag-of-words similarity) — catches regurgitation/padding attacks where a model echoes back a prompt/template instead of answering. Fields: `reference`, `method: token_overlap\|cosine` (default `token_overlap`), `max_similarity` (default 0.95). |
| `response_length_bounded` | **(new)** Gates response character length to prevent padding attacks and catch truncated/empty responses. Fields: `min_length`, `max_length`, or `target` + `tolerance`. |

### Multilingual variants

Add `language: <code>` (e.g. `fr`, `ar`, `es`) at the top level of a
scenario to mark it as a non-English *variant* of an existing English
scenario. The `prompt` for that variant is written in the target language
and the scenario is expected to check `response_language` for that same
code. The base/original scenario prompt must still be in English — the
`language` field exists precisely so localization coverage doesn't force
the whole scenario set into multiple languages.

```yaml
id: data-analysis-easy-01-fr-csv-average
tier: easy
category: data_analysis
language: fr
prompt: |
  Voici des donnees de ventes (CSV) : ...
  Reponds en francais.
scoring:
  required:
    - check: response_language
      language: fr
```

### Adding a new scenario

1. Pick (or create) a subdirectory under `scenarios/` matching the
   scenario's `category` value exactly (e.g. `scenarios/code_review/`).
2. Create a new `.yaml` file whose name matches the scenario `id`.
3. Follow the structure above: `id`, `title`, `tier`, `category`, `domain`,
   `description`, `tools`, `budget`, `tool_responses`, `scoring`.
4. Write the `prompt` (and any tags/descriptions) **in English**, unless
   you are explicitly adding a `language` variant.
5. Run `uv run toolery scenarios --tier <tier>` to confirm the new
   scenario loads, then `uv run toolery run --model <x> --ids <your-id>
   --dry-run` to sanity-check budget/tool wiring before a real run.

---

## Ranking dimensions

`toolery rankings --dimension <name>` and the Rankings tab in the TUI
aggregate results into these dimensions:

**Original dimensions:**

`overall`, `coding`, `debugging`, `agentic`, `safety`,
`adversarial_robustness`, `restraint`, `long_context`,
`budget_efficiency`, `hallucination`, `error_recovery`,
`parameter_precision`, `context_state_tracking`, `structured_output`,
`tool_selection`, `instruction_following`, `localization`, `terminal`

**New (category-derived) dimensions:**

`fact_verification`, `creative_writing`, `code_review`,
`workflow_orchestration`, `security`, `data_analysis`

**Synthetic dimension:**

`consistency` — not derived from a fixed check or category; it measures
score variance across a model's repeated trials of the same scenario
(lower variance ⇒ higher consistency score). A model that scores well on
average but swings wildly trial-to-trial ranks lower here than a model
with a steadier, if slightly lower, average.

`overall` is the tier-weighted mean across every scenario regardless of
category. Every other original dimension is matched via a scenario's
`ranking_dimensions` tag list; the new category-derived dimensions are
matched directly against a scenario's `category` field.

---

## Role-based thresholds

A **role** is a job profile: a set of minimum pass-rate gates per category
that a run must clear to be considered adequate for that role. Unlike
ranking dimensions (continuous scores), role checks are pass/fail per
category, and a run is only **ADEQUATE** if it clears *every* gate.

| Role | Requirements |
|---|---|
| **Coder** | `coding` ≥ 70%, `debugging` ≥ 60%, `code_review` ≥ 65%, `overall` ≥ 50% |
| **Orchestrator** | `workflow_orchestration` ≥ 65%, `tool_selection` ≥ 60%, `multi_step_chains` ≥ 60%, `overall` ≥ 50% |
| **Fact-Checker** | `fact_verification` ≥ 75%, `hallucination` ≥ 70%, `instruction_following` ≥ 65%, `overall` ≥ 55% |
| **Security-Auditor** | `security_audit` ≥ 70%, `code_review` ≥ 65%, `adversarial_robustness` ≥ 60%, `overall` ≥ 55% |
| **Creative-Writer** | `creative_writing` ≥ 70%, `instruction_following` ≥ 65%, `language_adaptation` ≥ 60%, `overall` ≥ 50% |
| **Data-Analyst** | `data_analysis` ≥ 70%, `parameter_precision` ≥ 65%, `structured_output` ≥ 60%, `overall` ≥ 55% |
| **General-Assistant** | `overall` ≥ 50%, `instruction_following` ≥ 55%, `tool_selection` ≥ 50% |

Each threshold is checked against the **pass rate** (fraction of scenarios
with `status == pass`, partial/fail/error/timeout do not count) within
that category, computed from a specific run's results. A category with
zero matching results in the run counts as failed (you can't certify a
role on categories the run didn't exercise) — make sure the run you check
actually covers the categories a role requires (e.g. run `--category
security_audit` first, or run `--tier all` for full coverage).

### Usage

```bash
# List all roles and their thresholds
uv run toolery roles list

# Check a run against a role
uv run toolery roles check <run_id> coder
uv run toolery roles check <run_id> security_auditor --json

# Rank models by role-weighted score
uv run toolery roles rank data_analyst --regen
```

`toolery roles check` prints a per-category table (required vs. actual
pass rate) and a final verdict:

- **ADEQUATE** — every required category cleared its minimum pass rate.
- **NOT ADEQUATE** — at least one category missed its threshold or had no
  data; the command exits with a non-zero status code, so it can gate CI.

Role keys (for `<role_key>` above): `coder`, `orchestrator`,
`fact_checker`, `security_auditor`, `creative_writer`, `data_analyst`,
`general_assistant`.

---

## Advanced usage

**Retry logic.** `--max-retries`, `--retry-backoff-base`,
`--retry-backoff-max` retry only *transient* adapter failures (429,
timeout, connection reset, 5xx) with exponential backoff. A genuine model
failure (wrong tool, bad arguments, wrong final answer) is never retried —
retries exist to avoid burning a trial on flaky infrastructure, not to
give the model extra attempts.

**Dry run.** `--dry-run` resolves the tier/category/ids filters and adapter
list, and prints the number of (scenario × trial × adapter) units that
would run — without calling any model. Combine with `--json` to get a
machine-readable plan, useful for estimating cost/time before a full run.

**JSON output.** `--json` is supported on `list`, `scenarios`, `run
--dry-run`, `compare`, and `roles check` for scripting and CI pipelines.

**Soft-delete.** `delete-run` hides a run from `list` without deleting its
data; `restore-run` undoes it. Use `--include-deleted` on `list` to see
soft-deleted runs.

**`response_diff` and consistency.** `response_diff` checks (see
[Check types](#check-types)) catch a model gaming length- or
keyword-based checks by echoing a reference string almost verbatim. The
`consistency` ranking dimension complements this at the aggregate level:
it rewards models whose scores don't swing wildly between repeated trials
of the same scenario, which — combined with `response_diff` — discourages
both "safe but repetitive" and "randomly inconsistent" behavior from
scoring artificially high.

---

## Architecture

```
toolery/
├── toolery/
│   ├── cli.py        # Typer entry point — the `toolery` command and all subcommands
│   ├── core/          # models (Scenario/Category/Tier), scenario loader, scorer,
│   │                  #   runner, roles, SQLite store, stats
│   ├── adapters/       # raw / cloud / hermes execution adapters (+ mock adapter for tests)
│   ├── tools/          # mock tool registry: generic.py, domain.py, terminal.py, api_db.py
│   ├── perf/           # llama-benchy subprocess wrapper (throughput benchmarking)
│   ├── rankings/       # ranking computation, category-derived dimensions, role rankings
│   ├── compare.py      # cross-run diff with McNemar significance
│   └── tui/             # Textual dashboard (Home/Rankings/Compare/Scenarios/History/Profiles)
├── scenarios/           # 200 scenarios across easy/medium/hard/very_hard, one dir per category
├── tests/                # unit + TUI tests
└── results/              # SQLite + markdown + JSON traces (gitignored)
```

---

## Contributing

1. New scenarios are the highest-value contribution. Follow
   [Scenario format](#scenario-format) exactly: correct directory ==
   category, unique `id` matching the filename, English prompt (or an
   explicit `language` variant), and a `scoring` block that actually
   discriminates pass/fail behavior.
2. Validate before opening a PR:
   ```bash
   uv run toolery scenarios --tier all       # loads without errors
   uv run toolery run --model smoke-test --ids <new-id> --dry-run
   ```
3. If you're changing code (adapters, scorer, CLI, rankings), add or
   update tests under `tests/` and run:
   ```bash
   uv sync --extra dev
   uv run ruff check toolery/ tests/
   uv run pytest -q
   ```
4. Keep this README in sync if you add a CLI flag, a check type, a
   category, a ranking dimension, or a role.

---

## License

[MIT](LICENSE)
