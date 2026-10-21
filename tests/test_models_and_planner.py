import pytest

from ai_harness.models import AgentRole, PlanDAG, PlanStep, TaskRequest
from ai_harness.planner import TaskRouter


def test_plan_dag_returns_topological_order():
    plan = PlanDAG(
        steps=[
            PlanStep(id="review", title="Review", role=AgentRole.REVIEWER, depends_on=["code"]),
            PlanStep(id="code", title="Code", role=AgentRole.CODER, depends_on=["scout"]),
            PlanStep(id="scout", title="Scout", role=AgentRole.SCOUT),
        ]
    )

    assert [step.id for step in plan.topological_order()] == ["scout", "code", "review"]


def test_plan_dag_rejects_cycle():
    with pytest.raises(ValueError, match="cycle"):
        PlanDAG(
            steps=[
                PlanStep(id="one", title="One", role=AgentRole.CODER, depends_on=["two"]),
                PlanStep(id="two", title="Two", role=AgentRole.CODER, depends_on=["one"]),
            ]
        )


def test_task_scope_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError, match="repository-relative"):
        TaskRequest(repository=str(tmp_path), goal="Make a safe change", file_scope=["../secret"])


def test_router_builds_full_acceptance_workflow(tmp_path):
    request = TaskRequest(repository=str(tmp_path), goal="Fix the calculator behavior")
    plan = TaskRouter().build_plan(request)

    assert [step.id for step in plan.topological_order()] == ["scout", "code", "verify", "review"]
