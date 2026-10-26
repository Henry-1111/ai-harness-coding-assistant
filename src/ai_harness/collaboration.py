from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

from .agents import AgentModel
from .models import (
    CodingDecision,
    CollaborationRoleAssignment,
    CollaborationTurn,
    FileEdit,
    RepairPlan,
    SearchHit,
    TaskRequest,
)
from .planner import TaskRouter


class CollaborationConflictError(ValueError):
    pass


class CollaborationScopeError(ValueError):
    pass


class CollaborationRoleError(RuntimeError):
    pass


class CollaborativeCoder:
    IMPLEMENTER_SYSTEM = """You are the implementation specialist in a bounded coding team.
Return JSON only: {"summary": string, "edits": [{"path": relative path,
"old_text": exact unique text or empty for a new file, "new_text": replacement,
"start_line": optional 1-based inclusive line, "end_line": optional 1-based inclusive line}]}.
Context snippets may show display-only line-number prefixes such as `12: `. Never include those
prefixes in old_text or new_text. For an existing file, copy old_text exactly from the code after
the prefix and provide the matching start_line and end_line shown in context. Include enough
surrounding code to make old_text unique. Never use empty old_text for an existing file.
Design the smallest complete implementation. Obey assignment.allowed_paths. When assignment.can_edit
is false, return an empty edits list. Do not execute commands or write outside file_scope."""

    TEST_SPECIALIST_SYSTEM = """You are the independent test specialist in a bounded coding team.
Without seeing another agent's proposal, identify the smallest test or integration edits required
by the goal and acceptance criteria. Return JSON only using the same {summary, edits} schema.
Remove display-only line-number prefixes from edit text and ensure each non-empty old_text is an
exact unique source excerpt. Preserve or correct the 1-based inclusive start_line and end_line for
existing files. Never use empty old_text to replace an existing file.
Return an empty edits list when the existing verification already covers the requested behavior.
Obey assignment.allowed_paths. When assignment.can_edit is false, return an empty edits list.
Never execute commands or write outside file_scope."""

    ARBITER_SYSTEM = """You are the patch arbiter in a bounded coding team.
Two independently generated proposals contain competing edits. Return one final conflict-free JSON
decision using the same {summary, edits} schema. Select the smallest change that satisfies the goal
and acceptance criteria. Do not combine overlapping replacements, invent paths, execute commands,
or write outside file_scope."""

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
        assignments = self.role_assignments(request, context)
        assignment_by_agent = {item.agent: item for item in assignments}
        shared = {
            "goal": request.goal,
            "acceptance_criteria": [item.model_dump() for item in request.acceptance_criteria],
            "context": [item.model_dump() for item in context],
            "previous_failure": previous_failure,
            "memory_context": memory_context or [],
            "repair_plan": repair_plan.model_dump(mode="json") if repair_plan else None,
        }
        role_payloads = {
            "implementer": {
                **shared,
                "file_scope": assignment_by_agent["implementer"].allowed_paths,
                "assignment": assignment_by_agent["implementer"].model_dump(mode="json"),
            },
            "test_specialist": {
                **shared,
                "file_scope": assignment_by_agent["test_specialist"].allowed_paths,
                "assignment": assignment_by_agent["test_specialist"].model_dump(
                    mode="json"
                ),
            },
        }
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai-harness-agent") as pool:
            implementation_future = pool.submit(
                copy_context().run,
                self.model.complete_json,
                system=self.IMPLEMENTER_SYSTEM,
                payload=role_payloads["implementer"],
            )
            test_future = pool.submit(
                copy_context().run,
                self.model.complete_json,
                system=self.TEST_SPECIALIST_SYSTEM,
                payload=role_payloads["test_specialist"],
            )
            futures = {
                "implementer": implementation_future,
                "test_specialist": test_future,
            }
            systems = {
                "implementer": self.IMPLEMENTER_SYSTEM,
                "test_specialist": self.TEST_SPECIALIST_SYSTEM,
            }
            proposals: Dict[str, CodingDecision] = {}
            role_attempts: Dict[str, int] = {}
            for role in ("implementer", "test_specialist"):
                role_attempts[role] = 1
                try:
                    proposal = CodingDecision.model_validate(futures[role].result())
                    self.validate_assignment(
                        proposal,
                        assignment_by_agent[role],
                    )
                except Exception as exc:
                    role_attempts[role] = 2
                    retry_payload = {
                        **role_payloads[role],
                        "previous_proposal_error": (
                            f"{type(exc).__name__}: {exc}"[-1000:]
                        ),
                        "retry_instruction": (
                            "Regenerate only this role's bounded proposal; preserve the other "
                            "role's completed work."
                        ),
                    }
                    try:
                        proposal = CodingDecision.model_validate(
                            self.model.complete_json(
                                system=systems[role],
                                payload=retry_payload,
                            )
                        )
                        self.validate_assignment(
                            proposal,
                            assignment_by_agent[role],
                        )
                    except Exception as retry_exc:
                        raise CollaborationRoleError(
                            f"{role} proposal failed after one isolated retry: {retry_exc}"
                        ) from retry_exc
                proposals[role] = proposal

        implementation = proposals["implementer"]
        test_proposal = proposals["test_specialist"]

        candidate_conflicts = self.detect_conflicts(
            implementation.edits + test_proposal.edits
        )
        arbitrated = bool(candidate_conflicts)
        if arbitrated:
            integrated = CodingDecision.model_validate(
                self.model.complete_json(
                    system=self.ARBITER_SYSTEM,
                    payload={
                        **shared,
                        "file_scope": request.file_scope,
                        "implementation_proposal": implementation.model_dump(mode="json"),
                        "test_proposal": test_proposal.model_dump(mode="json"),
                        "detected_conflicts": candidate_conflicts,
                    },
                )
            )
            self.validate_root_scope(integrated, request.file_scope)
        else:
            integrated = self.merge_proposals(implementation, test_proposal)
        unresolved = self.detect_conflicts(integrated.edits)
        if unresolved:
            raise CollaborationConflictError(
                f"integrated proposal still contains conflicts: {unresolved}"
            )
        return CollaborationTurn(
            round=round_number,
            implementation_proposal=implementation,
            test_proposal=test_proposal,
            integrated_decision=integrated,
            detected_conflicts=candidate_conflicts,
            participants=(
                ["implementer", "test_specialist", "arbiter"]
                if arbitrated
                else ["implementer", "test_specialist"]
            ),
            parallel_proposals=True,
            arbitrated=arbitrated,
            model_calls=sum(role_attempts.values()) + (1 if arbitrated else 0),
            assignments=assignments,
            role_attempts=role_attempts,
        )

    @staticmethod
    def role_assignments(
        request: TaskRequest,
        context: List[SearchHit],
    ) -> List[CollaborationRoleAssignment]:
        candidate_paths = list(request.file_scope)
        if not candidate_paths:
            candidate_paths = list(dict.fromkeys(hit.path for hit in context))
        implementation_paths, test_paths = TaskRouter.collaboration_scopes(
            candidate_paths
        )
        return [
            CollaborationRoleAssignment(
                agent="implementer",
                objective="Implement the production behavior inside the assigned paths",
                allowed_paths=implementation_paths,
                can_edit=bool(implementation_paths) or not candidate_paths,
            ),
            CollaborationRoleAssignment(
                agent="test_specialist",
                objective="Add or refine deterministic tests inside the assigned test paths",
                allowed_paths=test_paths,
                can_edit=bool(test_paths),
            ),
        ]

    @classmethod
    def validate_assignment(
        cls,
        proposal: CodingDecision,
        assignment: CollaborationRoleAssignment,
    ) -> None:
        if proposal.edits and not assignment.can_edit:
            raise CollaborationScopeError(
                f"{assignment.agent} is advisory-only because no writable paths were assigned"
            )
        for edit in proposal.edits:
            if assignment.allowed_paths and not cls.path_permitted(
                edit.path, assignment.allowed_paths
            ):
                raise CollaborationScopeError(
                    f"{assignment.agent} proposed an edit outside its assignment: {edit.path}"
                )

    @classmethod
    def validate_root_scope(cls, proposal: CodingDecision, paths: List[str]) -> None:
        for edit in proposal.edits:
            if paths and not cls.path_permitted(edit.path, paths):
                raise CollaborationScopeError(
                    f"arbiter proposed an edit outside the task scope: {edit.path}"
                )

    @staticmethod
    def path_permitted(path: str, allowed_paths: List[str]) -> bool:
        if "\\" in path:
            return False
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return False
        candidate_path = candidate.as_posix()
        for scope in allowed_paths:
            if "\\" in scope:
                continue
            boundary = PurePosixPath(scope)
            if boundary.is_absolute() or ".." in boundary.parts:
                continue
            scope_path = boundary.as_posix().rstrip("/")
            if candidate_path == scope_path or candidate_path.startswith(
                scope_path + "/"
            ):
                return True
        return False

    @staticmethod
    def merge_proposals(
        implementation: CodingDecision,
        test_proposal: CodingDecision,
    ) -> CodingDecision:
        edits: List[FileEdit] = []
        seen: set[tuple[str, str, int | None, int | None, str]] = set()
        for edit in implementation.edits + test_proposal.edits:
            identity = (
                edit.path,
                edit.old_text,
                edit.start_line,
                edit.end_line,
                edit.new_text,
            )
            if identity in seen:
                continue
            seen.add(identity)
            edits.append(edit)
        summaries = [implementation.summary]
        if test_proposal.edits:
            summaries.append(test_proposal.summary)
        return CodingDecision(summary="; ".join(summaries), edits=edits)

    @staticmethod
    def detect_conflicts(edits: List[FileEdit]) -> List[str]:
        replacements: Dict[tuple[str, str, int | None, int | None], str] = {}
        ranged: Dict[str, List[FileEdit]] = {}
        conflicts: List[str] = []
        for edit in edits:
            target = (edit.path, edit.old_text, edit.start_line, edit.end_line)
            previous = replacements.get(target)
            if previous is not None and previous != edit.new_text:
                conflicts.append(
                    f"{edit.path}: the same source text has competing replacements"
                )
            replacements[target] = edit.new_text
            if edit.start_line is None or edit.end_line is None:
                continue
            for previous_edit in ranged.setdefault(edit.path, []):
                assert previous_edit.start_line is not None
                assert previous_edit.end_line is not None
                overlaps = not (
                    edit.end_line < previous_edit.start_line
                    or edit.start_line > previous_edit.end_line
                )
                if overlaps and edit.new_text != previous_edit.new_text:
                    conflicts.append(
                        f"{edit.path}: line ranges overlap with competing replacements"
                    )
            ranged[edit.path].append(edit)
        return sorted(set(conflicts))
