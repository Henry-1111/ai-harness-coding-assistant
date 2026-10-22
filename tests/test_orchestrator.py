from pathlib import Path

import pytest
from conftest import ScriptedModel

from ai_harness.mcp import MCPToolCatalog, MCPToolCatalogConfig
from ai_harness.memory import MemoryLayer
from ai_harness.models import TaskRequest, TaskStatus
from ai_harness.orchestrator import ExecutionControl, HarnessOrchestrator
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


def test_execution_control_cancels_before_model_or_tools(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel([])
    orchestrator = HarnessOrchestrator(
        model, tmp_path / "data", runner_factory=LocalCommandSandbox
    )

    result = orchestrator.run(
        request_for(sample_repository),
        control=ExecutionControl(should_cancel=lambda: True),
    )

    assert result.status == TaskStatus.CANCELLED
    assert result.error == "task cancellation was requested"
    assert model.calls == []
    assert result.trace[-1].details["error_type"] == "task_cancelled"


def test_execution_control_enforces_task_deadline(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel([])
    orchestrator = HarnessOrchestrator(
        model, tmp_path / "data", runner_factory=LocalCommandSandbox
    )

    result = orchestrator.run(
        request_for(sample_repository),
        control=ExecutionControl(should_cancel=lambda: False, deadline=0),
    )

    assert result.status == TaskStatus.FAILED
    assert result.error == "task execution deadline exceeded"
    assert model.calls == []
    assert result.trace[-1].details["error_type"] == "task_timeout"


def test_full_workflow_isolated_verify_review_and_promote(tmp_path: Path, sample_repository: Path):
    model = ScriptedModel([successful_edit()])
    orchestrator = HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox)
    orchestrator.memory.append_bounded(
        MemoryLayer.PROJECT,
        orchestrator._project_id(str(sample_repository)),
        "experiences",
        {"task_id": "prior-task", "outcome": "succeeded", "repair_attempts": 0},
        unique_by="task_id",
    )

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
    assert model.calls[0]["payload"]["memory_context"][-1]["layer"] == "project_experience"
    assert orchestrator.memory.get(MemoryLayer.SESSION, result.task_id, "progress")["node"] == (
        "review"
    )
    assert "left - right" in (sample_repository / "calculator.py").read_text(encoding="utf-8")
    task_diff = orchestrator.diff(result.task_id)
    assert "-    return left - right" in task_diff
    assert "+    return left + right" in task_diff

    assert orchestrator.promote(result.task_id) == ["calculator.py"]
    assert "left + right" in (sample_repository / "calculator.py").read_text(encoding="utf-8")


def test_high_risk_task_waits_for_explicit_approval_before_execution(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel([successful_edit(), successful_edit()])
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
    )

    pending = orchestrator.run(
        request_for(
            sample_repository,
            goal="Modify production calculator behavior safely",
        )
    )

    assert pending.status == TaskStatus.NEEDS_INPUT
    assert pending.approval is not None
    assert pending.approval.status == "pending"
    assert pending.changed_files == []
    assert model.calls == []
    with pytest.raises(RuntimeError, match="explicit approval"):
        orchestrator.resume(pending.task_id)

    approved = orchestrator.decide_approval(
        pending.task_id,
        approved=True,
        actor="security-reviewer",
        note="Scoped production change approved",
    )

    assert approved.status == TaskStatus.SUCCEEDED
    assert approved.approval is not None
    assert approved.approval.status == "approved"
    assert approved.approval.decided_by == "security-reviewer"
    assert approved.changed_files == ["calculator.py"]
    assert len(model.calls) == 2
    assert approved.trace[0].node == "approval"

    replay = orchestrator.decide_approval(
        pending.task_id,
        approved=True,
        actor="security-reviewer",
    )
    assert replay == approved
    assert len(model.calls) == 2


def test_rejected_task_never_invokes_model_or_changes_source(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel([])
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
    )
    pending = orchestrator.run(
        request_for(
            sample_repository,
            goal="Delete production calculator data",
        )
    )

    rejected = orchestrator.decide_approval(
        pending.task_id,
        approved=False,
        actor="owner",
        note="Unsafe request",
    )

    assert rejected.status == TaskStatus.FAILED
    assert rejected.approval is not None
    assert rejected.approval.status == "rejected"
    assert rejected.error == "Approval rejected by owner: Unsafe request"
    assert rejected.changed_files == []
    assert model.calls == []
    assert "left - right" in (sample_repository / "calculator.py").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="already rejected"):
        orchestrator.decide_approval(
            pending.task_id,
            approved=True,
            actor="different-owner",
        )


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
    assert len(result.repair_history) == 1
    assert result.repair_history[0].can_retry is True
    assert model.calls[1]["payload"]["repair_plan"]["attempt"] == 1
    experiences = orchestrator.memory.get(
        MemoryLayer.PROJECT,
        orchestrator._project_id(str(sample_repository)),
        "experiences",
    )
    assert experiences[-1]["outcome"] == "succeeded"
    assert experiences[-1]["repair_attempts"] == 1


def test_repeated_failure_stops_self_healing_before_replan_budget(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel(
        [
            successful_edit(new="return left * right"),
            successful_edit(
                old="return left * right",
                new="return left * right  # still wrong",
            ),
        ]
    )
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
    )

    result = orchestrator.run(request_for(sample_repository, max_replans=3))

    assert result.status == TaskStatus.FAILED
    assert len(model.calls) == 2
    assert len(result.repair_history) == 2
    assert result.repair_history[-1].can_retry is False
    assert result.repair_history[-1].repeat_count == 1
    assert "self-healing stopped early" in (result.error or "")
    experiences = orchestrator.memory.get(
        MemoryLayer.PROJECT,
        orchestrator._project_id(str(sample_repository)),
        "experiences",
    )
    assert experiences[-1]["outcome"] == "failed"


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
    model = ScriptedModel([successful_edit(), successful_edit()])
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
        "implementation",
        "test_design",
        "code",
        "verify",
        "review",
    ]
    assert all(assignment.status == "completed" for assignment in state.assignments)
    assert len(result.collaboration_history) == 1
    assert result.collaboration_history[0].participants == [
        "implementer",
        "test_integrator",
    ]
    assert len(model.calls) == 2


def test_collaboration_can_be_disabled_for_complex_task(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel([successful_edit()])
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
        collaboration_mode="disabled",
    )

    result = orchestrator.run(
        request_for(
            sample_repository,
            goal="Refactor calculator architecture across multiple modules",
        )
    )

    assert result.status == TaskStatus.SUCCEEDED
    assert result.collaboration_history == []
    assert len(model.calls) == 1
    state = orchestrator.get(result.task_id)
    assert state is not None
    assert [assignment.step_id for assignment in state.assignments] == [
        "scout",
        "plan",
        "code",
        "verify",
        "review",
    ]


def test_collaboration_can_be_forced_for_fast_task(
    tmp_path: Path, sample_repository: Path
):
    model = ScriptedModel([successful_edit(), successful_edit()])
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
        collaboration_mode="forced",
    )

    result = orchestrator.run(request_for(sample_repository))

    assert result.status == TaskStatus.SUCCEEDED
    assert len(result.collaboration_history) == 1
    assert len(model.calls) == 2
    state = orchestrator.get(result.task_id)
    assert state is not None
    assert [assignment.step_id for assignment in state.assignments] == [
        "scout",
        "implementation",
        "test_design",
        "code",
        "verify",
        "review",
    ]


def test_insufficient_retrieval_can_enter_bounded_react_subgraph(
    tmp_path: Path, sample_repository: Path, monkeypatch
):
    monkeypatch.setenv("TEST_MCP_TOKEN", "Bearer secret")
    catalog = MCPToolCatalog(
        MCPToolCatalogConfig.model_validate(
            {
                "servers": {
                    "knowledge": {
                        "url": "http://localhost:9000/mcp",
                        "allowed_tenants": ["local"],
                        "headers_from_env": {"Authorization": "TEST_MCP_TOKEN"},
                    }
                },
                "tools": [
                    {
                        "name": "search_docs",
                        "server": "knowledge",
                        "remote_tool_name": "search",
                        "description": "search project documentation",
                        "allowed_tenants": ["local"],
                        "input_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ],
            }
        )
    )
    model = ScriptedModel(
        [
            {
                "action": "get_source_detail",
                "arguments": {"path": "calculator.py", "start_line": 1, "end_line": 2},
            },
            {"action": "finish", "answer": "The implementation defect is confirmed."},
            successful_edit(),
            successful_edit(),
        ]
    )
    orchestrator = HarnessOrchestrator(
        model,
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
        restricted_react_enabled=True,
        mcp_tool_catalog=catalog,
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
    assert {tool["name"] for tool in model.calls[0]["payload"]["tools"]} == {
        "get_source_detail",
        "search_docs",
        "search_project_knowledge",
    }
