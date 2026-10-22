from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents import AgentModel
from .models import (
    CodingDecision,
    CollaborationTurn,
    FileEdit,
    RepairPlan,
    SearchHit,
    TaskRequest,
)


class CollaborationConflictError(ValueError):
    pass


class CollaborativeCoder:
    IMPLEMENTER_SYSTEM = """You are the implementation specialist in a bounded coding team.
Return JSON only: {"summary": string, "edits": [{"path": relative path,
"old_text": exact unique text or empty for a new file, "new_text": replacement}]}.
Design the smallest complete implementation. Do not execute commands or write outside file_scope."""

    TEST_INTEGRATOR_SYSTEM = """You are the test/integration specialist in a bounded coding team.
Review the implementation proposal against the goal, acceptance criteria, repository context and
repair plan. Return one final conflict-free JSON decision using the same {summary, edits} schema.
Preserve correct implementation edits, add necessary tests when they are inside file_scope, and
remove conflicting or unjustified edits. Never emit commands."""

    def __init__(self, model: AgentModel) -> None:
        self.model = model

    def propose(
        self,
        request: TaskRequest,
        context: List[SearchHit],
        *,
        round_number: int,
        previous_failure: str = "",
        memory_context: Optional[List[Dict[str, Any]]] = None,
        repair_plan: Optional[RepairPlan] = None,
    ) -> CollaborationTurn:
        shared = {
            "goal": request.goal,
            "acceptance_criteria": [item.model_dump() for item in request.acceptance_criteria],
            "file_scope": request.file_scope,
            "context": [item.model_dump() for item in context],
            "previous_failure": previous_failure,
            "memory_context": memory_context or [],
            "repair_plan": repair_plan.model_dump(mode="json") if repair_plan else None,
        }
        implementation = CodingDecision.model_validate(
            self.model.complete_json(system=self.IMPLEMENTER_SYSTEM, payload=shared)
        )
        candidate_conflicts = self.detect_conflicts(implementation.edits)
        integrated = CodingDecision.model_validate(
            self.model.complete_json(
                system=self.TEST_INTEGRATOR_SYSTEM,
                payload={
                    **shared,
                    "implementation_proposal": implementation.model_dump(mode="json"),
                    "detected_conflicts": candidate_conflicts,
                },
            )
        )
        unresolved = self.detect_conflicts(integrated.edits)
        if unresolved:
            raise CollaborationConflictError(
                f"integrated proposal still contains conflicts: {unresolved}"
            )
        return CollaborationTurn(
            round=round_number,
            implementation_proposal=implementation,
            integrated_decision=integrated,
            detected_conflicts=candidate_conflicts,
        )

    @staticmethod
    def detect_conflicts(edits: List[FileEdit]) -> List[str]:
        replacements: Dict[tuple[str, str], str] = {}
        conflicts = []
        for edit in edits:
            target = (edit.path, edit.old_text)
            previous = replacements.get(target)
            if previous is not None and previous != edit.new_text:
                conflicts.append(
                    f"{edit.path}: the same source text has competing replacements"
                )
            replacements[target] = edit.new_text
        return sorted(set(conflicts))
