from __future__ import annotations

from .models import AgentRole, PlanDAG, PlanStep, TaskRequest


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

    def requires_plan(self, request: TaskRequest) -> bool:
        text = request.goal.lower()
        return bool(request.acceptance_criteria) or any(
            marker in text for marker in self.COMPLEXITY_MARKERS
        )

    def build_plan(self, request: TaskRequest) -> PlanDAG:
        acceptance = [item.description for item in request.acceptance_criteria]
        return PlanDAG(
            steps=[
                PlanStep(
                    id="scout",
                    title="Locate the smallest relevant code context",
                    role=AgentRole.SCOUT,
                    outputs=["repo_map", "context"],
                    acceptance=["Relevant files and symbols are identified"],
                ),
                PlanStep(
                    id="code",
                    title="Implement the requested change",
                    role=AgentRole.CODER,
                    depends_on=["scout"],
                    inputs=["context", "goal"],
                    outputs=["patch"],
                    acceptance=acceptance or ["Requested behavior is implemented"],
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
