"""Role-based rankings — rank (model, adapter) pairs by a weighted score
derived from a Role's category weights (toolery.core.roles.Role).

Unlike the per-dimension rankings in rankings/compute.py (which use scenario
`ranking_dimensions` tags and tier weighting with time-decay across runs),
role rankings are intentionally simple: they pool ALL of a pair's stored
results, bucket by scenario `category`, compute a pass rate per category,
and combine them with the role's weight multipliers. This mirrors the
"does this model qualify for this job" framing of toolery.core.roles.check_role,
just applied across every model instead of one run.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from toolery.core.roles import Role, RoleCheckResult, category_pass_rates, get_role
from toolery.core.store import Store


@dataclass(frozen=True)
class RoleRankingRow:
    model: str
    adapter: str
    weighted_score: float
    n_results: int
    category_scores: dict[str, tuple[float, int]]  # category -> (pass_rate, n)


def compute_role_ranking(store: Store, role_key: str) -> list[RoleRankingRow]:
    """Rank every (model, adapter) pair that has data by its role-weighted score.

    weighted_score = sum(weight[cat] * pass_rate[cat] for cat in weights present)
                      / sum(weight[cat] for cat in weights present)

    Only categories the pair actually has results for contribute (both to the
    numerator and denominator) — a pair untested on a weighted category is
    simply not penalized/rewarded for it here (use check_role() against a
    specific run for a strict pass/fail gate that treats missing data as fail).

    Raises KeyError for an unknown role_key.
    """
    role = get_role(role_key)
    if role is None:
        raise KeyError(f"unknown role: {role_key!r}")

    runs = {r["run_id"]: r for r in store.fetch_all_runs()}
    with store.conn() as c:
        all_results = [dict(r) for r in c.execute("SELECT * FROM scenario_results").fetchall()]

    pair_results: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        meta = runs.get(r["run_id"])
        if not meta:
            continue
        pair_results[(meta["model"], r["adapter"])].append(r)

    rows: list[RoleRankingRow] = []
    for (model, adapter), results in pair_results.items():
        rates = category_pass_rates(results)
        num = 0.0
        den = 0.0
        for cat, weight in role.weights.items():
            rate_n = rates.get(cat)
            if rate_n is None:
                continue
            rate, _n = rate_n
            num += weight * rate
            den += weight
        if den <= 0:
            continue
        rows.append(RoleRankingRow(
            model=model, adapter=adapter, weighted_score=num / den,
            n_results=len(results), category_scores=rates,
        ))

    rows.sort(key=lambda r: -r.weighted_score)
    return rows


def render_role_ranking_md(role: Role, rows: list[RoleRankingRow]) -> str:
    """Render a markdown table for a role ranking (mirrors ranking.md.j2's shape)."""
    lines = [
        f"# {role.name} Ranking",
        "",
        f"_{role.description}_",
        "",
        "Weighted score combines each category's pass rate using the role's "
        "weight multipliers: " + ", ".join(
            f"{cat}×{w:g}" for cat, w in role.weights.items()
        ) + ".",
        "",
        "| # | Model | Adapter | Weighted Score | Results |",
        "|---|-------|---------|----------------:|--------:|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row.model} | {row.adapter} | "
            f"{row.weighted_score * 100:.1f}% | {row.n_results} |"
        )
    lines.append("")
    lines.append("## Required thresholds for this role")
    lines.append("")
    lines.append("| Category | Min pass rate |")
    lines.append("|----------|---------------:|")
    for req in role.required:
        lines.append(f"| {req.category} | {req.min_pass_rate * 100:.0f}% |")
    lines.append("")
    return "\n".join(lines) + "\n"


def regenerate_role_rankings(*, store: Store, out_dir) -> None:
    """Compute and write role_<key>.md for every defined role into out_dir."""
    from toolery.core.roles import list_roles

    out_dir.mkdir(parents=True, exist_ok=True)
    for role in list_roles():
        rows = compute_role_ranking(store, role.key)
        md = render_role_ranking_md(role, rows)
        (out_dir / f"role_{role.key}.md").write_text(md, encoding="utf-8")


def render_role_check_md(result: RoleCheckResult) -> str:
    """Render the pass/fail table + verdict for a single run's role check
    (used by `toolery roles check` for a human-readable summary)."""
    verdict = "ADEQUATE" if result.adequate else "NOT ADEQUATE"
    lines = [
        f"# Role check: {result.role.name} — run {result.run_id}",
        "",
        f"_{result.role.description}_",
        "",
        "| Category | Required | Actual | Result |",
        "|----------|---------:|-------:|:------:|",
    ]
    for c in result.checks:
        actual = "n/a" if c.actual_pass_rate is None else f"{c.actual_pass_rate * 100:.1f}% (n={c.n})"
        mark = "✓" if c.passed else "✗"
        lines.append(f"| {c.category} | {c.min_pass_rate * 100:.0f}% | {actual} | {mark} |")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    return "\n".join(lines)
