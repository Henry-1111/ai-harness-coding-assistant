from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    AcceptanceCriterion,
    RiskLevel,
    RouteKind,
    TaskComplexity,
    TaskRequest,
    TaskType,
)
from .planner import TaskRouter


class PlanningBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    goal: str = Field(min_length=5, max_length=4000)
    acceptance_criteria: List[str] = Field(default_factory=list)
    file_scope: List[str] = Field(default_factory=list)
    verification_commands: List[str] = Field(default_factory=lambda: ["pytest -q"])
    expected_task_type: TaskType
    expected_complexity: TaskComplexity
    expected_risk: RiskLevel
    expected_route: RouteKind
    split: str = "test"


class PlanningCaseResult(BaseModel):
    id: str
    task_type_correct: bool
    complexity_correct: bool
    risk_correct: bool
    route_correct: bool
    planning_decision_correct: bool
    plan_structurally_valid: bool
    actual_task_type: TaskType
    actual_complexity: TaskComplexity
    actual_risk: RiskLevel
    actual_route: RouteKind
    plan_steps: List[str]
    errors: List[str] = Field(default_factory=list)


class PlanningBenchmarkReport(BaseModel):
    dataset_sha256: str
    cases: int
    task_type_accuracy: float
    complexity_accuracy: float
    risk_accuracy: float
    route_accuracy: float
    planning_decision_accuracy: float
    plan_structural_validity_rate: float
    failures: List[PlanningCaseResult]
    details: List[PlanningCaseResult]


def load_planning_cases(path: Path) -> List[PlanningBenchmarkCase]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("cases"), list):
        value = value["cases"]
    elif not isinstance(value, list):
        raise ValueError("planning dataset must be a JSON array or an object with cases")
    cases = [PlanningBenchmarkCase.model_validate(item) for item in value]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("planning benchmark case ids must be unique")
    goals = [" ".join(case.goal.lower().split()) for case in cases]
    if len(goals) != len(set(goals)):
        raise ValueError("planning benchmark goals must be unique")
    return cases


def planning_cases_sha256(cases: List[PlanningBenchmarkCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class PlanningEvaluator:
    """Measure deterministic task normalization, routing and generated Plan integrity."""

    def __init__(self, router: TaskRouter | None = None) -> None:
        self.router = router or TaskRouter()

    def evaluate(
        self, cases: List[PlanningBenchmarkCase], *, split: str | None = None
    ) -> PlanningBenchmarkReport:
        dataset_hash = planning_cases_sha256(cases)
        selected = [case for case in cases if split is None or case.split == split]
        if not selected:
            raise ValueError("at least one planning benchmark case is required")
        details = [self._evaluate_case(case) for case in selected]
        count = len(details)
        failures = [item for item in details if item.errors]
        return PlanningBenchmarkReport(
            dataset_sha256=dataset_hash,
            cases=count,
            task_type_accuracy=self._rate(details, "task_type_correct"),
            complexity_accuracy=self._rate(details, "complexity_correct"),
            risk_accuracy=self._rate(details, "risk_correct"),
            route_accuracy=self._rate(details, "route_correct"),
            planning_decision_accuracy=self._rate(details, "planning_decision_correct"),
            plan_structural_validity_rate=self._rate(details, "plan_structurally_valid"),
            failures=failures,
            details=details,
        )

    def _evaluate_case(self, case: PlanningBenchmarkCase) -> PlanningCaseResult:
        request = TaskRequest(
            repository=".",
            goal=case.goal,
            acceptance_criteria=[
                AcceptanceCriterion(description=value)
                for value in case.acceptance_criteria
            ],
            file_scope=case.file_scope,
            verification_commands=case.verification_commands,
        )
        normalized = self.router.normalizer.normalize(request)
        route = self.router.decide(normalized)
        collaboration = self.router.collaboration_policy(request, normalized, route)
        plan = self.router.build_plan(
            request,
            normalized,
            route,
            collaboration=collaboration.enabled,
        )
        ordered = plan.topological_order()
        assignments = self.router.assignments(plan)
        step_ids = [step.id for step in ordered]
        errors: List[str] = []

        checks = {
            "task_type_correct": normalized.task_type == case.expected_task_type,
            "complexity_correct": normalized.complexity == case.expected_complexity,
            "risk_correct": normalized.risk_level == case.expected_risk,
            "route_correct": route.route == case.expected_route,
            "planning_decision_correct": route.requires_planning
            == (case.expected_route != RouteKind.FAST),
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(name)

        structurally_valid = self._plan_is_valid(
            step_ids,
            assignments=[assignment.step_id for assignment in assignments],
            requires_planning=route.requires_planning,
            collaborative=collaboration.enabled,
        )
        if not structurally_valid:
            errors.append("plan_structurally_valid")
        return PlanningCaseResult(
            id=case.id,
            **checks,
            plan_structurally_valid=structurally_valid,
            actual_task_type=normalized.task_type,
            actual_complexity=normalized.complexity,
            actual_risk=normalized.risk_level,
            actual_route=route.route,
            plan_steps=step_ids,
            errors=errors,
        )

    @staticmethod
    def _plan_is_valid(
        steps: List[str],
        *,
        assignments: List[str],
        requires_planning: bool,
        collaborative: bool,
    ) -> bool:
        required = {"scout", "code", "verify", "review"}
        if not required.issubset(steps) or assignments != steps:
            return False
        if len(steps) != len(set(steps)):
            return False
        if ("plan" in steps) != requires_planning:
            return False
        collaboration_steps = {"implementation", "test_design"}
        if collaboration_steps.issubset(steps) != collaborative:
            return False
        positions = {step: index for index, step in enumerate(steps)}
        return (
            positions["scout"] < positions["code"] < positions["verify"] < positions["review"]
        )

    @staticmethod
    def _rate(items: List[PlanningCaseResult], field: str) -> float:
        return round(sum(bool(getattr(item, field)) for item in items) / len(items), 6)
