import pytest

from ai_harness.models import (
    AgentRole,
    CollaborationPolicy,
    CollaborationTurn,
    PlanDAG,
    PlanStep,
    RiskLevel,
    RouteKind,
    TaskComplexity,
    TaskRequest,
    TaskType,
)
from ai_harness.planner import TaskNormalizer, TaskRouter


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


def test_normalizer_classifies_intent_complexity_risk_and_clarification(tmp_path):
    request = TaskRequest(
        repository=str(tmp_path),
        goal="  Refactor authentication across multiple modules in production  ",
        verification_commands=["pytest -q", "ruff check ."],
    )

    normalized = TaskNormalizer().normalize(request)

    assert normalized.normalized_goal == (
        "Refactor authentication across multiple modules in production"
    )
    assert normalized.task_type == TaskType.REFACTOR
    assert normalized.complexity == TaskComplexity.COMPLEX
    assert normalized.risk_level == RiskLevel.CRITICAL
    assert normalized.requires_human_approval is True
    assert normalized.requires_clarification is True


def test_complex_route_inserts_plan_node_and_assigns_every_step(tmp_path):
    request = TaskRequest(
        repository=str(tmp_path),
        goal="Refactor the service architecture across multiple modules",
        file_scope=["src"],
    )
    router = TaskRouter()
    normalized = router.normalizer.normalize(request)
    route = router.decide(normalized)
    plan = router.build_plan(request, normalized, route)
    assignments = router.assignments(plan)

    assert route.route == RouteKind.COMPLEX
    assert route.requires_planning is True
    assert [step.id for step in plan.topological_order()] == [
        "scout",
        "plan",
        "implementation",
        "test_design",
        "integration",
        "code",
        "verify",
        "review",
    ]
    assert [assignment.step_id for assignment in assignments] == [
        "scout",
        "plan",
        "implementation",
        "test_design",
        "integration",
        "code",
        "verify",
        "review",
    ]


def test_collaboration_policy_uses_explicit_complexity_signals(tmp_path):
    router = TaskRouter()
    single_file = TaskRequest(
        repository=str(tmp_path),
        goal="Refactor calculator implementation",
        file_scope=["calculator.py"],
    )
    normalized = router.normalizer.normalize(single_file)
    route = router.decide(normalized)

    policy = router.collaboration_policy(single_file, normalized, route)

    assert route.route == RouteKind.COMPLEX
    assert policy.enabled is False
    assert policy.strategy == "single"

    cross_file = single_file.model_copy(
        update={
            "goal": "Refactor calculator across multiple modules",
            "file_scope": ["calculator.py", "service.py"],
        }
    )
    normalized = router.normalizer.normalize(cross_file)
    route = router.decide(normalized)
    policy = router.collaboration_policy(cross_file, normalized, route)

    assert policy.enabled is True
    assert policy.strategy == "implementer_reviewer"
    assert policy.reasons == ["cross_module_goal", "multi_path_scope"]
    assert policy.max_rounds == 2
    assert policy.model_call_budget == 6


def test_collaboration_policy_forced_and_disabled_modes(tmp_path):
    request = TaskRequest(
        repository=str(tmp_path),
        goal="Fix calculator behavior",
        max_replans=3,
    )
    router = TaskRouter()
    normalized = router.normalizer.normalize(request)
    route = router.decide(normalized)

    forced = router.collaboration_policy(request, normalized, route, "forced")
    disabled = router.collaboration_policy(request, normalized, route, "disabled")

    assert forced.enabled is True
    assert forced.max_rounds == 2
    assert forced.model_call_budget == 6
    assert disabled == CollaborationPolicy(
        reasons=["collaboration explicitly disabled"]
    )


def test_collaboration_policy_rejects_inconsistent_budget():
    with pytest.raises(ValueError, match="cover both agents"):
        CollaborationPolicy(
            enabled=True,
            strategy="implementer_reviewer",
            max_rounds=2,
            model_call_budget=3,
        )


def test_collaboration_turn_loads_legacy_serialized_state():
    decision = {"summary": "legacy proposal", "edits": []}

    turn = CollaborationTurn.model_validate(
        {
            "round": 1,
            "implementation_proposal": decision,
            "integrated_decision": decision,
            "participants": ["implementer", "test_integrator"],
        }
    )

    assert turn.test_proposal is None
    assert turn.parallel_proposals is False
    assert turn.arbitrated is False
