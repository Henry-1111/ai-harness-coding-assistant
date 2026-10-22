from pathlib import Path

import pytest
from conftest import ScriptedModel

from ai_harness.collaboration import CollaborationConflictError, CollaborativeCoder
from ai_harness.models import TaskRequest


def decision(edits):
    return {"summary": "proposal", "edits": edits}


def edit(new_text: str):
    return {
        "path": "calculator.py",
        "old_text": "return left - right",
        "new_text": new_text,
    }


def test_test_integrator_receives_candidate_and_resolves_conflicts(tmp_path: Path):
    model = ScriptedModel(
        [
            decision([edit("return left + right"), edit("return sum((left, right))")]),
            decision([edit("return left + right")]),
        ]
    )

    turn = CollaborativeCoder(model).propose(
        TaskRequest(
            repository=str(tmp_path),
            goal="Refactor calculator architecture across multiple modules",
            file_scope=["calculator.py"],
        ),
        [],
        round_number=1,
    )

    assert len(turn.detected_conflicts) == 1
    assert turn.integrated_decision.edits[0].new_text == "return left + right"
    assert model.calls[1]["payload"]["detected_conflicts"] == turn.detected_conflicts


def test_unresolved_integrator_conflict_is_rejected(tmp_path: Path):
    conflict = decision([edit("return left + right"), edit("return sum((left, right))")])
    model = ScriptedModel([conflict, conflict])

    with pytest.raises(CollaborationConflictError, match="still contains conflicts"):
        CollaborativeCoder(model).propose(
            TaskRequest(
                repository=str(tmp_path),
                goal="Refactor calculator architecture across multiple modules",
                file_scope=["calculator.py"],
            ),
            [],
            round_number=1,
        )
