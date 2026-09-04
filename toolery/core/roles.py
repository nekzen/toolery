"""Role-based threshold profiles.

Each Role names a job profile (Coder, Orchestrator, ...) and specifies the
minimum per-category pass rate a model must clear to be considered adequate
for that role, plus a set of weight multipliers used to build a weighted
score for role-based rankings (see toolery/rankings/roles.py).

Categories referenced by `required`/`weights` are scenario `category` values
(see toolery.core.models.Category) with one synthetic addition: "overall",
which means "pass rate across every scenario regardless of category".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoleRequirement:
    """One minimum-pass-rate gate for a role.

    `category` is either "overall" (all scenarios) or a scenario category
    value (e.g. "coding", "security_audit").
    """
    category: str
    min_pass_rate: float


@dataclass(frozen=True)
class Role:
    key: str
    name: str
    description: str
    required: list[RoleRequirement]
    weights: dict[str, float] = field(default_factory=dict)


ROLES: dict[str, Role] = {
    "coder": Role(
        key="coder",
        name="Coder",
        description="Writes and fixes code — needs strong coding, debugging, and review skills.",
        required=[
            RoleRequirement("coding", 0.70),
            RoleRequirement("debugging", 0.60),
            RoleRequirement("code_review", 0.65),
            RoleRequirement("overall", 0.50),
        ],
        weights={"coding": 3.0, "debugging": 2.0, "code_review": 2.0, "overall": 1.0},
    ),
    "orchestrator": Role(
        key="orchestrator",
        name="Orchestrator",
        description="Coordinates multi-step tool workflows — needs strong planning and tool selection.",
        required=[
            RoleRequirement("workflow_orchestration", 0.65),
            RoleRequirement("tool_selection", 0.60),
            RoleRequirement("multi_step_chains", 0.60),
            RoleRequirement("overall", 0.50),
        ],
        weights={
            "workflow_orchestration": 3.0, "tool_selection": 2.0,
            "multi_step_chains": 2.0, "overall": 1.0,
        },
    ),
    "fact_checker": Role(
        key="fact_checker",
        name="Fact-Checker",
        description="Verifies claims and grounds answers — needs low hallucination and precise instruction-following.",
        required=[
            RoleRequirement("fact_verification", 0.75),
            RoleRequirement("hallucination", 0.70),
            RoleRequirement("instruction_following", 0.65),
            RoleRequirement("overall", 0.55),
        ],
        weights={
            "fact_verification": 3.0, "hallucination": 2.5,
            "instruction_following": 1.5, "overall": 1.0,
        },
    ),
    "security_auditor": Role(
        key="security_auditor",
        name="Security-Auditor",
        description="Audits code and systems for vulnerabilities — needs security and code-review depth.",
        required=[
            RoleRequirement("security_audit", 0.70),
            RoleRequirement("code_review", 0.65),
            RoleRequirement("adversarial_robustness", 0.60),
            RoleRequirement("overall", 0.55),
        ],
        weights={
            "security_audit": 3.0, "code_review": 2.0,
            "adversarial_robustness": 2.0, "overall": 1.0,
        },
    ),
    "creative_writer": Role(
        key="creative_writer",
        name="Creative-Writer",
        description="Produces styled, audience-adapted prose — needs instruction-following and tone control.",
        required=[
            RoleRequirement("creative_writing", 0.70),
            RoleRequirement("instruction_following", 0.65),
            RoleRequirement("language_adaptation", 0.60),
            RoleRequirement("overall", 0.50),
        ],
        weights={
            "creative_writing": 3.0, "instruction_following": 2.0,
            "language_adaptation": 1.5, "overall": 1.0,
        },
    ),
    "data_analyst": Role(
        key="data_analyst",
        name="Data-Analyst",
        description="Extracts and structures insight from data — needs precision and structured output.",
        required=[
            RoleRequirement("data_analysis", 0.70),
            RoleRequirement("parameter_precision", 0.65),
            RoleRequirement("structured_output", 0.60),
            RoleRequirement("overall", 0.55),
        ],
        weights={
            "data_analysis": 3.0, "parameter_precision": 2.0,
            "structured_output": 2.0, "overall": 1.0,
        },
    ),
    "general_assistant": Role(
        key="general_assistant",
        name="General-Assistant",
        description="Broad, dependable everyday helper — needs solid overall competence, no deep specialty.",
        required=[
            RoleRequirement("overall", 0.50),
            RoleRequirement("instruction_following", 0.55),
            RoleRequirement("tool_selection", 0.50),
        ],
        weights={"overall": 2.0, "instruction_following": 1.5, "tool_selection": 1.5},
    ),
}


def get_role(key: str) -> Role | None:
    """Look up a role profile by its key. Returns None for unknown keys."""
    return ROLES.get(key)


def list_roles() -> list[Role]:
    """All defined roles, in declaration order."""
    return list(ROLES.values())


@dataclass(frozen=True)
class RequirementCheck:
    category: str
    min_pass_rate: float
    actual_pass_rate: float | None  # None when there is no data for this category
    n: int  # number of results contributing to actual_pass_rate
    passed: bool


@dataclass(frozen=True)
class RoleCheckResult:
    role: Role
    run_id: str
    checks: list[RequirementCheck]

    @property
    def adequate(self) -> bool:
        """ADEQUATE iff every requirement passed (missing data counts as fail)."""
        return bool(self.checks) and all(c.passed for c in self.checks)


def category_pass_rates(results: list[dict]) -> dict[str, tuple[float, int]]:
    """Group scenario_results rows by `category` and compute (pass_rate, n)
    per category, plus a synthetic "overall" bucket covering every row.

    A result "passes" when status == 'pass' (partial/fail/error/timeout do
    not count), matching the semantics used elsewhere for the "Passed x/N"
    ranking column.
    """
    buckets: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category") or "unknown"
        buckets.setdefault(cat, []).append(r)
        buckets.setdefault("overall", []).append(r)
    out: dict[str, tuple[float, int]] = {}
    for cat, rows in buckets.items():
        n = len(rows)
        passed = sum(1 for r in rows if (r.get("status") or "") == "pass")
        out[cat] = (passed / n if n else 0.0, n)
    return out


def check_role(store, run_id: str, role_key: str) -> RoleCheckResult:
    """Evaluate whether a run meets a role's minimum-pass-rate thresholds.

    Raises KeyError if role_key is unknown. A category with zero matching
    results in the run yields actual_pass_rate=None and passed=False (you
    can't certify a role on categories the run didn't exercise).
    """
    role = ROLES.get(role_key)
    if role is None:
        raise KeyError(f"unknown role: {role_key!r} (known: {sorted(ROLES)})")

    results = store.fetch_results_for_run(run_id)
    rates = category_pass_rates(results)

    checks: list[RequirementCheck] = []
    for req in role.required:
        rate_n = rates.get(req.category)
        if rate_n is None:
            checks.append(RequirementCheck(
                category=req.category, min_pass_rate=req.min_pass_rate,
                actual_pass_rate=None, n=0, passed=False,
            ))
            continue
        rate, n = rate_n
        checks.append(RequirementCheck(
            category=req.category, min_pass_rate=req.min_pass_rate,
            actual_pass_rate=rate, n=n, passed=rate >= req.min_pass_rate,
        ))
    return RoleCheckResult(role=role, run_id=run_id, checks=checks)
