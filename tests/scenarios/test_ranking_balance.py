# tests/scenarios/test_ranking_balance.py
"""Asserts dimension-count balance after the W1-W4 backlog is in place.

Reference: docs/superpowers/specs/2026-05-26-ranking-balance-design.md
"""
from __future__ import annotations

import collections
from pathlib import Path

from toolery.core.scenario import load_all_scenarios

# Expected counts after Phase 2 (fact_verification, creative_writing,
# code_review, workflow_orchestration, security_audit, data_analysis — 57
# new scenarios, some with fr/ar/es localized variants). Baseline before
# Phase 2 was 143 (see git history for the pre-Phase-2 snapshot of this
# file with the debugging-era counts).
EXPECTED_COUNTS = {
    "overall": 200,
    "agentic": 52,
    "coding": 42,
    "debugging": 16,
    "safety": 36,
    "adversarial_robustness": 10,
    "terminal": 14,
    "budget_efficiency": 20,
    "parameter_precision": 18,
    "restraint": 16,
    "hallucination": 38,
    "tool_selection": 20,
    "long_context": 13,
    "error_recovery": 15,
    "structured_output": 16,
    "context_state_tracking": 16,
    "instruction_following": 30,
    "localization": 29,
}

# Tier counts keep the original hand-designed balance (40/45/34/24) for the
# pre-Phase-2 scenario set; Phase 2 added 57 scenarios across all 4 tiers
# using the same ~2/2/2/1-2 per-category split, shifting the totals here.
EXPECTED_TIER_COUNTS = {
    "easy": 60,
    "medium": 63,
    "hard": 47,
    "very_hard": 30,
}


def _aggregate():
    root = Path(__file__).resolve().parents[2] / "scenarios"
    scenarios = load_all_scenarios(root)
    by_dim: collections.Counter = collections.Counter()
    by_tier: collections.Counter = collections.Counter()
    for s in scenarios:
        for d in s.ranking_dimensions:
            by_dim[d] += 1
        by_tier[s.tier.value] += 1
    return by_dim, by_tier


def test_dimension_counts_match_balance_spec():
    by_dim, _ = _aggregate()
    diffs = {k: (by_dim[k], v) for k, v in EXPECTED_COUNTS.items() if by_dim[k] != v}
    assert not diffs, f"dimension count drift: {diffs}"


def test_tier_counts_match_balance_spec():
    _, by_tier = _aggregate()
    diffs = {k: (by_tier[k], v) for k, v in EXPECTED_TIER_COUNTS.items() if by_tier[k] != v}
    assert not diffs, f"tier count drift: {diffs}"
