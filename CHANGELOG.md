# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-05

First release of the nekzen fork. Everything below diverges from
karol.palys' toolery at 0.4.1 (commit `36c8c0c`); versions 0.4.1 and
earlier are the original author's work.

### Added
- **6 new scenario categories, 57 scenarios**: `fact_verification`,
  `creative_writing`, `code_review`, `workflow_orchestration`,
  `security_audit`, `data_analysis` — including French, Spanish, and Arabic
  localization variants (Phase 2).
- **Category-derived ranking dimensions** (`fact_verification`,
  `creative_writing`, `code_review`, `workflow_orchestration`, `security`,
  `data_analysis`) matched on the scenario `category` column, plus the
  synthetic `consistency` dimension (score variance across trials) (Phase 3).
- **Role-based thresholds**: 7 job profiles (Coder, Orchestrator,
  Fact-Checker, Security-Auditor, Creative-Writer, Data-Analyst,
  General-Assistant) with per-category minimum pass-rate gates and an
  ADEQUATE / NOT ADEQUATE verdict — `toolery roles list|check|rank` (Phase 3).
- **Profiles tab: role viability board** — one row per model (latest run),
  one ✓/✗ column per role, with a `✗·gap` marker distinguishing "below
  threshold" from "required category not covered by that run".
- New checks: `response_diff` (near-duplicate/regurgitation detection) and
  `response_length_bounded` (padding/truncation gate) (Phase 1).
- CLI: `--json` output (`list`, `scenarios`, `run --dry-run`, `compare`,
  `roles check`), `--dry-run` planning, transient-failure retry
  (`--max-retries` + exponential backoff), run soft-delete/restore
  (`delete-run` / `restore-run`) (Phase 1).
- Use-case personas can now weight the category-derived dimensions; all 7
  personas gained weights for the 6 new dimensions (17 → 23 each).
- `rankings --dimension` validates its value and lists the valid dimensions.

### Fixed
- `response_language` could never detect French or Spanish (the detector
  only knew pl/en/de), making all 12 FR/ES scenarios unpassable; fr/es
  marker sets and diacritic hints added.
- `_pattern_found` now folds diacritics on both sides, so ASCII-authored
  patterns ("requete preparee") match accented responses ("requête préparée").
- Scenario audit: bracketed a malformed emoji character class, scored the
  previously-unverified 5th constraint of the multi-constraint story,
  word-bounded substring traps (`then`, `PUT`, `growing`, `false`), fixed
  self-contradicting forbidden patterns, acrostic letter-order check,
  Arabic-Indic numeral acceptance, and a missing boundary-case guard in the
  three-way-branch workflow scenario.
- Creative-Writer role required a `language_adaptation` category that no
  scenario has ever carried — the role could never be ADEQUATE. Gate removed
  (localization remains its own ranking dimension).
- Windows portability: `encoding="utf-8"` on every bare
  `read_text()`/`write_text()` call site — the cp1252 default broke 8 tests
  and crashed trace-writing on any emoji in a model response.
- TUI drift: rankings columns now derive from `STANDARD_DIMENSIONS`; run
  profile folds category-derived dimensions in; `raw` adapter filter added
  to Profiles; octa cluster rendering in Compare; perf/L10n legend
  corrections.
- `run_benchy` prefers the venv-installed (version-locked) `llama-benchy`
  over the `uvx` fallback, which always fetches the latest PyPI release.

### Removed
- Four phantom `Category` enum members (`structured_reasoning`,
  `toolset_scale`, `autonomous_planning`, `creative_composition`) — no
  scenario at any revision ever used them.
- One-off scripts tied to the original author's hardware and runs:
  `rescore_minimax.py`, `test.sh`, `scripts/codex_self_eval.py`.

### Changed
- `golden_probe.py` and `patch_timeouts_merge.py` moved to `scripts/`;
  `patch_timeouts_merge.py` honors `TOOLERY_RESULTS_DIR`.
- Project metadata: repository URLs point to the fork; nekzen added as
  maintainer (karol.palys remains the author).
- README rewritten and re-verified against the codebase (categories, roles,
  scoring semantics, environment variables, install extras).

## [0.4.1] - 2026-06-16

### Added
- Rankings tab: the rank (`#`) and `Model` columns are now frozen — they stay
  pinned and visible while the wide matrix scrolls horizontally (`Adapter` and
  everything to its right still scroll).
- Scenarios tab: English **Difficulty** and **Category** filter dropdowns that
  narrow the scenario picker, plus a Task-panel metadata header showing the
  live (empirically re-tiered) difficulty, category and tags. The picker is now
  ordered by current tier instead of legacy folder order, so difficulties no
  longer look jumbled.

### Fixed
- Rankings tab: horizontal scroll no longer snaps back to the leftmost column on
  the 5s poll — columns are built once and only the rows refresh.
- Rankings tab: the 5s poll no longer rebuilds the table on nearly every tick.
  The change-detection signature now keys on discrete data signals (run count,
  most-recent-run pass counts, non-decayed perf, cluster, stale flag) instead of
  the continuously time-decayed score floats, eliminating the periodic jitter.
- Rankings tab: the selected row's highlight is preserved across a genuine
  rebuild (rows carry a stable `model|adapter|cluster` key) instead of snapping
  back to the top row.
- Rankings tab: the frozen columns now show the normal per-row zebra striping
  instead of a flat blue (`datatable--fixed`) highlight.

## [0.4.0] - 2026-06-12

### Added
- Rankings tab: new "Passed" column (right after Overall) showing raw
  passed-trial counts, e.g. `512/715`. Counts come from the pair's MOST RECENT
  run only (a single run's fraction is meaningful; a decay-blend across runs is
  not) and count full passes — partials don't. Sortable (by fraction, since
  totals can differ across scenario-set versions); `model_mean` view sums
  counts across adapters; legend entry included. Backed by a new `pass_counts`
  field on `compute_matrix()` rows.
- Cross-tool state gates for mock rules: `if_tool_called` / `if_tool_not_called`
  on `ToolResponseRule` make a rule conditional on whether another tool has been
  invoked earlier in the trial (self-references count strictly prior calls).
  Enables honest stateful sims: tests that fail until `edit_file` happened,
  `git_status` that reflects a prior `git_commit`, a tests/ listing that shows a
  file only after `write_file`.

### Changed
- Default `--timeout-scale` is now 2.0 (CLI and `Runner`). The scale-1.0
  per-scenario budgets were tuned for fast local serving at low concurrency and
  killed reasoning models mid-first-answer (MiMo audit 2026-06-12: 37 timeout
  trials with zero recorded messages; passes of the same scenarios landed at
  96-100% of the limit). Explicit `--timeout-scale` still overrides.
- `very-hard-19-branch-memory` redesigned: a dedicated `deploy(env, version)`
  tool replaces the expectation that models invent the literal shell command
  `deploy --env staging --version v2.3.1` (0% pass across all models — it
  measured command divination, not state tracking). bash_exec remains as a fair
  distractor steering to the deploy tool; scoring now also requires the turn-2
  git_status to happen before the deploy. The state-tracking intent (remember
  staging vs prod across turns) is unchanged.
- The remaining four 0%-across-all-models scenarios (8 models × 5 trials, runs
  of 2026-06-11/12) repaired:
  - `easy-16`: orientation commands (`ls`/`pwd`/`find`/`which`) return plausible
    output instead of "command not found" — every audited model opened with
    `ls -la` to orient and spiraled on the absurd error; the listing now also
    hints at pytest (pytest.ini, tests/).
  - `easy-32`: prompt now states the no-tools constraint ("Without using any
    tools or lookups…"). Previously the schema OFFERED get_stock_price with
    zero signal that using it was the failing path — the scenario punished
    exactly what every other scenario rewards. Refusal phrase list broadened
    with staleness disclosures (knowledge cutoff, real-time, may be outdated).
  - `very-hard-13`: budget 1→2 (search-the-date then learn-it's-non-numeric was
    an instant kill); dropped the forbidden 43.2x check and none_of number
    traps that zeroed the ideal answer ("ill-typed; if you mean the year,
    sqrt(1867)≈43.21"); flagging-phrase list broadened. A bare number with no
    flagging still fails.
  - `very-hard-21`: budget 1→2 so inspecting before a destructive kill —
    textbook ops practice — is no longer an instant kill; inspection commands
    (ss/netstat/lsof/ps) return a listing with a second postgres on :5433 as
    the don't-touch-others trap; kill-by-discovered-pid accepted as precise;
    brittle literal response check ("lsof -ti :5432", spacing-sensitive)
    replaced with regex/semantic checks. Name-based blanket kills still fail.

### Fixed
- Mock/grader artifacts found while auditing the MiMo-V2.5 run (2026-06-12);
  previously these zeroed trials on model behavior that was correct or defensible:
  - `easy-06`: `get_contacts` only matched `{name: "Bob"}` exactly — a full-name
    query ("Bob Reilly") returned `[]` and made the scenario unpassable
    (pass-rate 0% across 3 models). Now regex-matched.
  - `hard-01`: `run_tests` returned FAILED on `call_index: 0` regardless of state,
    so a model that fixed the bug before running tests saw a false FAILED on
    correct code. Now gated on `edit_file` having happened (read_file likewise).
  - `very-hard-01`, `hard-02`, `very-hard-04`: `read_file` returned the literal
    placeholder `"(file content)"` for every path; models that actually read code
    before editing got garbage and probed past budget. Real abridged contents added.
    `very-hard-01` `run_bash` now steers to dedicated tools instead of silently
    returning empty success.
  - `medium-01`: `git_status` was stateless and kept reporting the file as
    untracked after a correct commit, provoking a verification spiral. Now stateful.
  - `medium-27`: `grep -c`/`--count` (a fully correct way to count) fell through
    to the line-dump rule; now returns the count.
  - `medium-31`/`medium-32`: `list_files` returned the workspace-root listing for
    every path including `tests/`; now path-aware (and stateful in medium-32).
  - `easy-31`/`hard-21`: the blanket destructive-block also rejected read-only
    inspection (`ls`, `find` without `-delete`, `test`, `pwd`), punishing exactly
    the cautious behavior these scenarios reward. Read-only commands now succeed;
    destructive ones remain blocked.
  - `medium-36`: clarification grader keyword list broadened (EN+PL) — a model
    that asked a perfectly good clarifying question ("I need a bit more
    information…") failed on phrasing alone.

## [0.3.0] - 2026-06-11

### Changed
- Empirical re-tiering of all 143 scenarios. Tiers are now assigned by measured
  pass-rate across three models (MiniMax-M2.7-AWQ, Nex-N2-Pro-W4A16, Qwen3.6):
  scenarios ranked easiest→hardest and sliced into the existing balanced quotas
  (40/45/34/24) via quantile assignment. Only the `tier:` field changed; scenario
  ids keep their historical prefix as a stable key, so an id like `easy-39` may
  now legitimately sit in a different tier.

### Added
- `toolery run --ids <a,b,...>` runs an explicit scenario subset.
- `golden_probe.py`: replays a hand-authored ideal play through the live mock +
  scorer to prove a scenario is solvable (passability guard).
- Standard competition ranking ("1224") for tied scores in both the generated
  `.md` rankings and the TUI matrix podium: tied models share a medal and the
  next distinct score skips ahead (two 100% → both gold, next bronze).

### Fixed
- Grader robustness to model output style: unicode-folded + markdown-stripped
  phrase matching, integer patterns match the integer part of decimals,
  digit-leading patterns are token-bounded, numeric CSV cells compared by value,
  and structured-output unwrapping tolerates a prose preamble / multiple fenced
  blocks (takes the last). Eliminates a class of false-negatives where correct
  answers failed on presentation.
- Mock runtime supports a generic `<param>_regex` matcher (not just
  `command_regex`); `get_weather_global` gained optional `date`/`units` params
  and lost a misleading description. Several scenarios were unpassable before
  these fixes (e.g. `query_contains` was never a real matcher).
- TUI: scenario display name decoupled from tier — the name drops its tier
  prefix and the tier column is sourced from the scenario definition for every
  row (pending/running included), so name and tier no longer contradict.

## [0.2.0] - 2026-06-04

### Added
- History tab auto-refreshes: a 3s poll plus an on-show hook surface freshly
  completed runs without pressing Ctrl+R or restarting the app. The poll is a
  no-op unless the run set actually changed, and it preserves the active filter
  and the highlighted row across rebuilds.
- `--timeout-scale` option on `toolery run` (and the `Runner`): multiplies each
  scenario's `timeout_seconds`, so slow cloud/reasoning endpoints (high RTT +
  long chain-of-thought) aren't killed mid-answer. Also honored from run config.
- Cluster (DGX Spark topology) is now a first-class ranking axis: the same
  model+adapter on `single`/`dual`/`triple`/`quad`/`octa` shows as separate
  rows; re-runs of the same configuration still aggregate (decay-weighted mean
  + run-to-run stability).
- Scenarios tab now shows the task content (title, description, prompt) above
  the per-model results.
- GitHub project scaffolding: MIT `LICENSE`, CI workflow (ruff + pytest),
  `CONTRIBUTING.md`, and this changelog.

### Changed
- Overall ranking score is now tier-weighted only: every dimension counts
  equally (weight 1.0) when aggregating the Overall column, matching what the
  TUI legend and markdown reports already described. Per-dimension weighting is
  reserved for use-case persona columns, which keep their own weight maps.
- Renamed the project `LLM-test` → **Toolery**: Python package `toolery`,
  CLI command `toolery`, and environment variables `TOOLERY_*`
  (`TOOLERY_BASE_URL`, `TOOLERY_RESULTS_DIR`, …).
- Execution adapters trimmed to `raw` / `cloud` / `hermes` (removed the
  `claude_code` and `codex` CLI harnesses).
- Renamed the last TUI tab `Setup` → `Profiles` to match its content.

### Fixed
- Rankings table no longer resets horizontal scroll on the 5s poll: the
  render-skip signature ignores sub-display-precision decay drift, and scroll
  position is restored across rebuilds.
- `octa` (8-spark) runs now render and validate correctly across the rankings
  and launch flow.
- Corrected the "training control" label in the Home tab run-in-progress panel.

## [0.1.0]

- Initial benchmark: 4-tier scenario suite, deterministic scoring, Textual TUI,
  14-dimension ranking system, ASCII + PNG charts, and llama-benchy perf
  integration.
