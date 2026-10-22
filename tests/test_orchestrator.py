from pathlib import Path

from conftest import ScriptedModel

from ai_harness.memory import MemoryLayer
from ai_harness.models import TaskRequest, TaskStatus
from ai_harness.orchestrator import HarnessOrchestrator
from ai_harness.sandbox import LocalCommandSandbox


def request_for(repository: Path, **updates) -> TaskRequest:
    values = {
        "repository": str(repository),
        "goal": "Fix calculator add so two plus three equals five",
        "file_scope": ["calculator.py"],
        "verification_commands": ["python -m pytest -q"],
    }
    values.update(updates)
    return TaskRequest(**values)


def successful_edit(old="return left - right", new="return left + right"):
    return {
        "summary": "Correct addition implementation",
        "edits": [{"path": "calculator.py", "old_text": old, "new_text": new}],
    }


def test_full_workflow_isolated_verify_review_and_promote(tmp_path: Path, sample_repository: Path):
    model = ScriptedModel([successful_edit()])
    orchestrator = HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox)

    result = orchestrator.run(request_for(sample_repository))

    assert result.status == TaskStatus.SUCCEEDED
    assert result.changed_files == ["calculator.py"]
    assert result.verification[0].exit_code == 0
    assert result.retrieval_trace is not None
    assert result.retrieval_trace.stages[-1].stage == "judge"
    assert [event.node for event in result.trace] == ["scout", "code", "verify", "review"]
    assert result.route is not None
    assert result.route.route.value == "fast"
    assert [milestone.code for milestone in result.milestones] == [
        "CP1",
        "CP2",
        "CP3",
        "CP4",
        "CP5",
        "CP6",
        "CP7",
        "CP8",
    ]
    assert model.calls[0]["payload"]["memory_context"][0]["layer"] == "policy"
    assert orchestrator.memory.get(MemoryLayer.SESSION, result.task_id, "progress")["node"] == (
        "review"
    )
    assert "left - right" in (sample_repository / "calculator.py").read_text(encoding="utf-8")
    task_diff = orchestrator.diff(result.task_id)
    assert "-    return left - right" in task_diff
    assert "+    return left + right" in task_diff

    assert orchestrator.promote(result.task_id) == ["calculator.py"]
    assert "left + right" in (sample_repository / "calculator.py").read_text(encoding="utf-8")


def test_failed_verification_triggers_bounded_replan(tmp_path: Path, sample_repository: Path):
    model = ScriptedModel(
        [
            successful_edit(new="return left * right"),
            successful_edit(old="return left * right", new="return left + right"),
        ]
    )
    orchestrator = HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox)

    result = orchestrator.run(request_for(sample_repository, max_replans=1))

    assert result.status == TaskStatus.SUCCEEDED
    assert len(model.calls) == 2
    state = orchestrator.get(result.task_id)
    assert state is not None
    assert state.replan_count == 1


def test_failed_task_cannot_be_promoted(tmp_path: Path, sample_repository: Path):
    model = ScriptedModel([successful_edit(new="return left * right")])
    orchestrator = HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox)
    result = orchestrator.run(request_for(sample_repository, max_replans=0))

    assert result.status == TaskStatus.FAILED
    try:
        orchestrator.promote(result.task_id)
    except RuntimeError as exc:
        assert "approved" in str(exc)
    else:
        raise AssertionError("failed task should not be promoted")


def test_resume_of_completed_task_is_idempotent(tmp_path: Path, sample_repository: Path):
    model = ScriptedModel([successful_edit()])
    orchestrator = HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox)
    result = orchestrator.run(request_for(sample_repository))

    replay = orchestrator.resume(result.task_id)

    assert replay == result
    assert len(model.calls) == 1


def test_complex_task_executes_plan_node_and_records_assignments(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel([successful_edit()])
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
    )

    result = orchestrator.run(
        request_for(
            sample_repository,
            goal="Refactor calculator architecture across multiple modules",
        )
    )

    assert result.status == TaskStatus.SUCCEEDED
    assert result.route is not None
    assert result.route.route.value == "complex"
    assert [event.node for event in result.trace] == [
        "scout",
        "plan",
        "code",
        "verify",
        "review",
    ]
    state = orchestrator.get(result.task_id)
    assert state is not None
    assert [assignment.step_id for assignment in state.assignments] == [
        "scout",
        "plan",
        "code",
        "verify",
        "review",
    ]
    assert all(assignment.status == "completed" for assignment in state.assignments)


def test_insufficient_retrieval_can_enter_bounded_react_subgraph(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel(
        [
            {
                "action": "get_source_detail",
                "arguments": {"path": "calculator.py", "start_line": 1, "end_line": 2},
            },
            {"action": "finish", "answer": "The implementation defect is confirmed."},
            successful_edit(),
        ]
    )
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
        restricted_react_enabled=True,
    )

    result = orchestrator.run(
        request_for(
            sample_repository,
            goal="Refactor calculator architecture across multiple modules",
        )
    )

    assert result.status == TaskStatus.SUCCEEDED
    state = orchestrator.get(result.task_id)
    assert state is not None
    react = state.scratchpad["restricted_react"]
    assert react["tool_calls"] == 1
    assert react["stop_reason"] == "model_finished"
    assert any(hit.source == "restricted_react" for hit in state.context)
