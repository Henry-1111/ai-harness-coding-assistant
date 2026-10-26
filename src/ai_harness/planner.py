from __future__ import annotations

import re
from typing import Literal, Optional

from .models import (
    AgentAssignment,
    AgentRole,
    CollaborationPolicy,
    NormalizedTask,
    PlanDAG,
    PlanStep,
    RiskLevel,
    RouteDecision,
    RouteKind,
    TaskComplexity,
    TaskRequest,
    TaskType,
)


class TaskNormalizer:
    """Normalize a request into explicit intent, risk and acceptance metadata."""

    TYPE_MARKERS = {
        TaskType.MIGRATION: ("migration", "migrate", "迁移", "schema"),
        TaskType.REFACTOR: ("refactor", "重构", "architecture", "架构"),
        TaskType.BUGFIX: ("fix", "bug", "error", "failure", "修复", "错误", "异常"),
        TaskType.TEST: ("test", "coverage", "测试", "覆盖率"),
        TaskType.FEATURE: ("add", "implement", "feature", "新增", "实现", "功能"),
    }
    COMPLEX_MARKERS = (
        "architecture",
        "migration",
        "refactor",
        "multiple",
        "across",
        "架构",
        "迁移",
        "重构",
        "多个",
        "跨模块",
        "全链路",
    )
    HIGH_RISK_MARKERS = (
        "auth",
        "permission",
        "security",
        "payment",
        "quota",
        "database",
        "migration",
        "鉴权",
        "权限",
        "安全",
        "支付",
        "扣款",
        "数据库",
        "迁移",
    )
    CRITICAL_MARKERS = (
        "production",
        "drop table",
        "delete data",
        "rotate secret",
        "生产环境",
        "删除数据",
        "清空数据",
        "密钥轮换",
    )
    VAGUE_MARKERS = ("optimize", "improve", "better", "优化一下", "改进一下", "弄一下")

    def normalize(self, request: TaskRequest) -> NormalizedTask:
        normalized_goal = re.sub(r"\s+", " ", request.goal).strip()
        lowered = normalized_goal.lower()
        task_type = TaskType.GENERAL
        for candidate, markers in self.TYPE_MARKERS.items():
            if any(marker in lowered for marker in markers):
                task_type = candidate
                break

        complexity_score = 0
        complexity_score += 2 if any(marker in lowered for marker in self.COMPLEX_MARKERS) else 0
        complexity_score += 1 if len(request.file_scope) > 2 else 0
        complexity_score += 1 if len(request.acceptance_criteria) > 1 else 0
        complexity_score += 1 if len(request.verification_commands) > 1 else 0
        if task_type in {TaskType.MIGRATION, TaskType.REFACTOR}:
            complexity_score += 1
        complexity = (
            TaskComplexity.COMPLEX
            if complexity_score >= 3
            else TaskComplexity.STANDARD
            if complexity_score >= 1
            else TaskComplexity.SIMPLE
        )

        if any(marker in lowered for marker in self.CRITICAL_MARKERS):
            risk = RiskLevel.CRITICAL
        elif any(marker in lowered for marker in self.HIGH_RISK_MARKERS):
            risk = RiskLevel.HIGH
        elif complexity != TaskComplexity.SIMPLE:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        clarification_questions = []
        if (
            any(marker in lowered for marker in self.VAGUE_MARKERS)
            and not request.acceptance_criteria
        ):
            clarification_questions.append("What observable behavior defines success?")
        if not request.file_scope and complexity == TaskComplexity.COMPLEX:
            clarification_questions.append("Which repository paths are allowed to change?")

        acceptance = [item.description for item in request.acceptance_criteria]
        if not acceptance:
            acceptance = [
                "The requested behavior is implemented",
                "All configured verification commands pass",
                "Changes remain inside the declared file scope",
            ]
        return NormalizedTask(
            normalized_goal=normalized_goal,
            task_type=task_type,
            complexity=complexity,
            risk_level=risk,
            derived_acceptance=acceptance,
            requires_clarification=bool(clarification_questions),
            clarification_questions=clarification_questions,
            requires_human_approval=risk in {RiskLevel.HIGH, RiskLevel.CRITICAL},
        )


class TaskRouter:
    """Turn a natural-language coding request into a bounded execution route."""

    COMPLEXITY_MARKERS = (
        "refactor",
        "migration",
        "architecture",
        "multiple",
        "across",
        "重构",
        "迁移",
        "多个",
        "跨",
    )
    COLLABORATION_MARKERS = (
        "multiple",
        "across",
        "cross-module",
        "cross module",
        "多个",
        "跨模块",
        "跨文件",
        "全链路",
    )
    TEST_PATH_MARKERS = (
        "_test.",
        "/tests/",
        "/test/",
        ".test.",
        ".spec.",
    )

    def __init__(self, normalizer: Optional[TaskNormalizer] = None) -> None:
        self.normalizer = normalizer or TaskNormalizer()

    def decide(self, normalized: NormalizedTask) -> RouteDecision:
        reasons = [
            f"task_type={normalized.task_type.value}",
            f"complexity={normalized.complexity.value}",
            f"risk={normalized.risk_level.value}",
        ]
        if normalized.risk_level == RiskLevel.CRITICAL:
            route = RouteKind.HUMAN_GATE
        elif (
            normalized.complexity == TaskComplexity.COMPLEX
            or normalized.risk_level == RiskLevel.HIGH
        ):
            route = RouteKind.COMPLEX
        elif normalized.complexity == TaskComplexity.STANDARD:
            route = RouteKind.STANDARD
        else:
            route = RouteKind.FAST
        requires_planning = route != RouteKind.FAST
        roles = [AgentRole.SCOUT]
        if requires_planning:
            roles.append(AgentRole.PLANNER)
        roles.extend([AgentRole.CODER, AgentRole.REVIEWER])
        return RouteDecision(
            route=route,
            reasons=reasons,
            requires_planning=requires_planning,
            required_roles=roles,
        )

    def requires_plan(self, request: TaskRequest) -> bool:
        normalized = self.normalizer.normalize(request)
        return self.decide(normalized).requires_planning

    def collaboration_policy(
        self,
        request: TaskRequest,
        normalized: NormalizedTask,
        route: RouteDecision,
        mode: Literal["auto", "disabled", "forced"] = "auto",
    ) -> CollaborationPolicy:
        if mode == "disabled":
            return CollaborationPolicy(
                reasons=["collaboration explicitly disabled"],
            )

        max_rounds = min(2, request.max_replans + 1)
        if mode == "forced":
            return CollaborationPolicy(
                enabled=True,
                strategy="implementer_reviewer",
                reasons=["collaboration explicitly forced"],
                max_rounds=max_rounds,
                model_call_budget=max_rounds * 5,
            )

        lowered = normalized.normalized_goal.lower()
        cross_module = any(marker in lowered for marker in self.COLLABORATION_MARKERS)
        multi_path_scope = (
            len(request.file_scope) > 1 and route.route != RouteKind.FAST
        )
        broad_unbounded_scope = not request.file_scope and route.route in {
            RouteKind.COMPLEX,
            RouteKind.HUMAN_GATE,
        }
        high_risk = normalized.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
        migration = normalized.task_type == TaskType.MIGRATION
        signals = {
            "cross_module_goal": cross_module,
            "multi_path_scope": multi_path_scope,
            "broad_unbounded_scope": broad_unbounded_scope,
            "high_risk": high_risk,
            "migration": migration,
        }
        reasons = [name for name, present in signals.items() if present]
        enabled = bool(reasons)
        if not enabled:
            return CollaborationPolicy(
                reasons=[
                    "single-agent route: no cross-module, multi-path, migration or high-risk signal"
                ]
            )
        return CollaborationPolicy(
            enabled=True,
            strategy="implementer_reviewer",
            reasons=reasons,
            max_rounds=max_rounds,
            model_call_budget=max_rounds * 5,
        )

    def build_plan(
        self,
        request: TaskRequest,
        normalized: Optional[NormalizedTask] = None,
        route: Optional[RouteDecision] = None,
        collaboration: Optional[bool] = None,
    ) -> PlanDAG:
        normalized = normalized or self.normalizer.normalize(request)
        route = route or self.decide(normalized)
        steps = [
            PlanStep(
                id="scout",
                title="Locate the smallest relevant code context",
                role=AgentRole.SCOUT,
                outputs=["repo_map", "context"],
                acceptance=["Relevant files and symbols are identified"],
            )
        ]
        code_dependency = "scout"
        if route.requires_planning:
            steps.append(
                PlanStep(
                    id="plan",
                    title="Validate the execution DAG, risk and acceptance evidence",
                    role=AgentRole.PLANNER,
                    depends_on=["scout"],
                    inputs=["normalized_task", "repo_map", "context"],
                    outputs=["validated_plan", "assignments"],
                    acceptance=["The DAG is acyclic and every step has an owner"],
                )
            )
            code_dependency = "plan"
        collaboration = (
            self.collaboration_policy(request, normalized, route).enabled
            if collaboration is None
            else collaboration
        )
        if collaboration:
            implementation_paths, test_paths = self.collaboration_scopes(
                request.file_scope
            )
            steps.extend(
                [
                    PlanStep(
                        id="implementation",
                        title="Produce an implementation proposal",
                        role=AgentRole.CODER,
                        depends_on=[code_dependency],
                        inputs=["context", "goal"],
                        outputs=["implementation_proposal"],
                        acceptance=["Proposal is scoped and structurally valid"],
                        allowed_paths=implementation_paths,
                        can_edit=bool(implementation_paths) or not request.file_scope,
                    ),
                    PlanStep(
                        id="test_design",
                        title="Independently propose tests and integration safeguards",
                        role=AgentRole.REVIEWER,
                        depends_on=[code_dependency],
                        inputs=["context", "goal", "acceptance_criteria"],
                        outputs=["test_proposal"],
                        acceptance=["Test proposal is scoped and independently derived"],
                        allowed_paths=test_paths,
                        can_edit=bool(test_paths),
                    ),
                    PlanStep(
                        id="integration",
                        title="Merge proposals or arbitrate conflicting edits",
                        role=AgentRole.REVIEWER,
                        depends_on=["implementation", "test_design"],
                        inputs=["implementation_proposal", "test_proposal"],
                        outputs=["integrated_decision"],
                        acceptance=["Final proposal is conflict-free and test-aware"],
                    ),
                ]
            )
            code_dependency = "integration"
        steps.extend(
            [
                PlanStep(
                    id="code",
                    title="Implement the requested change",
                    role=AgentRole.CODER,
                    depends_on=[code_dependency],
                    inputs=["context", "goal"],
                    outputs=["patch"],
                    acceptance=normalized.derived_acceptance,
                ),
                PlanStep(
                    id="verify",
                    title="Run deterministic verification",
                    role=AgentRole.CODER,
                    depends_on=["code"],
                    inputs=["patch", "verification_commands"],
                    outputs=["verification_results"],
                    acceptance=["All configured commands exit successfully"],
                ),
                PlanStep(
                    id="review",
                    title="Review scope, evidence and acceptance criteria",
                    role=AgentRole.REVIEWER,
                    depends_on=["verify"],
                    inputs=["diff", "verification_results"],
                    outputs=["review_decision"],
                    acceptance=["Reviewer approves the bounded change"],
                ),
            ]
        )
        return PlanDAG(steps=steps)

    @staticmethod
    def assignments(plan: PlanDAG) -> list[AgentAssignment]:
        return [
            AgentAssignment(
                assignment_id=f"assignment-{index}",
                step_id=step.id,
                role=step.role,
                objective=step.title,
                allowed_paths=step.allowed_paths,
                can_edit=step.can_edit,
            )
            for index, step in enumerate(plan.topological_order(), start=1)
        ]

    @classmethod
    def collaboration_scopes(cls, paths: list[str]) -> tuple[list[str], list[str]]:
        test_paths = [path for path in paths if cls.is_test_path(path)]
        implementation_paths = [path for path in paths if path not in test_paths]
        return implementation_paths, test_paths

    @classmethod
    def is_test_path(cls, path: str) -> bool:
        lowered = f"/{path.lower()}"
        name = path.rsplit("/", 1)[-1].lower()
        return name.startswith("test_") or any(
            marker in lowered for marker in cls.TEST_PATH_MARKERS
        )
