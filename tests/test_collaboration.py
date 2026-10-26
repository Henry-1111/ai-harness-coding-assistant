import json
from pathlib import Path
from threading import Barrier, Lock

import pytest

from ai_harness.agents import OpenAICompatibleModel, capture_task_model_usage
from ai_harness.collaboration import CollaborationConflictError, CollaborativeCoder
from ai_harness.models import FileEdit, TaskRequest


def decision(edits):
    return {"summary": "proposal", "edits": edits}


def edit(new_text: str):
    return {
        "path": "calculator.py",
        "old_text": "return left - right",
        "new_text": new_text,
    }


class RoleModel:
    def __init__(self, implementation, test_proposal, arbitration=None):
        self.implementation = implementation
        self.test_proposal = test_proposal
        self.arbitration = arbitration
        self.calls = []
        self.lock = Lock()

    def complete_json(self, *, system, payload):
        with self.lock:
            self.calls.append({"system": system, "payload": payload})
        if "implementation specialist" in system:
            return self.implementation
        if "independent test specialist" in system:
            return self.test_proposal
        if "patch arbiter" in system and self.arbitration is not None:
            return self.arbitration
        raise AssertionError("unexpected role")


def test_parallel_proposals_merge_without_model_arbitration(tmp_path: Path):
    model = RoleModel(
        decision([edit("return left + right")]),
        decision(
            [
                {
                    "path": "test_calculator.py",
                    "old_text": "",
                    "new_text": "def test_add():\n    assert add(2, 3) == 5\n",
                }
            ]
        ),
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

    assert turn.detected_conflicts == []
    assert turn.parallel_proposals is True
    assert turn.arbitrated is False
    assert turn.model_calls == 2
    assert turn.participants == ["implementer", "test_specialist"]
    assert {item.path for item in turn.integrated_decision.edits} == {
        "calculator.py",
        "test_calculator.py",
    }
    assert len(model.calls) == 2


def test_conflicting_parallel_proposals_use_arbiter(tmp_path: Path):
    model = RoleModel(
        decision([edit("return left + right")]),
        decision([edit("return sum((left, right))")]),
        decision([edit("return left + right")]),
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

    assert turn.arbitrated is True
    assert turn.model_calls == 3
    assert turn.participants == ["implementer", "test_specialist", "arbiter"]
    assert turn.integrated_decision.edits[0].new_text == "return left + right"
    assert model.calls[-1]["payload"]["detected_conflicts"] == turn.detected_conflicts


def test_unresolved_arbiter_conflict_is_rejected(tmp_path: Path):
    conflict = decision([edit("return left + right"), edit("return sum((left, right))")])
    model = RoleModel(conflict, decision([]), conflict)

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


def test_overlapping_line_ranges_are_detected():
    conflicts = CollaborativeCoder.detect_conflicts(
        [
            FileEdit.model_validate({
                "path": "calculator.py",
                "old_text": "line one",
                "new_text": "first",
                "start_line": 2,
                "end_line": 4,
            }),
            FileEdit.model_validate({
                "path": "calculator.py",
                "old_text": "line two",
                "new_text": "second",
                "start_line": 4,
                "end_line": 5,
            }),
        ]
    )

    assert conflicts == [
        "calculator.py: line ranges overlap with competing replacements"
    ]


def test_parallel_model_calls_preserve_task_usage(monkeypatch, tmp_path: Path):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"summary": "no extra edits", "edits": []}
                            )
                        }
                    }
                ],
            }

    monkeypatch.setattr(
        "ai_harness.agents.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    model = OpenAICompatibleModel(
        "http://127.0.0.1:8000/v1",
        "temporary-key",
        "test-model",
    )

    with capture_task_model_usage() as usage:
        turn = CollaborativeCoder(model).propose(
            TaskRequest(
                repository=str(tmp_path),
                goal="Refactor across multiple modules",
            ),
            [],
            round_number=1,
        )

    snapshot = usage.snapshot()
    assert turn.model_calls == 2
    assert snapshot.calls == 2
    assert snapshot.prompt_tokens == 24
    assert snapshot.completion_tokens == 8
    assert snapshot.total_tokens == 32


def test_implementer_and_test_specialist_execute_concurrently(tmp_path: Path):
    barrier = Barrier(2)

    class BlockingRoleModel:
        def complete_json(self, *, system, payload):
            assert "specialist" in system
            barrier.wait(timeout=1)
            return {"summary": "parallel proposal", "edits": []}

    turn = CollaborativeCoder(BlockingRoleModel()).propose(
        TaskRequest(
            repository=str(tmp_path),
            goal="Refactor across multiple modules",
        ),
        [],
        round_number=1,
    )

    assert turn.parallel_proposals is True
    assert turn.model_calls == 2
