import json
from pathlib import Path

import pytest

from ai_harness.models import RiskLevel, RouteKind, TaskComplexity, TaskType
from ai_harness.planning_evaluation import (
    PlanningBenchmarkCase,
    PlanningEvaluator,
    load_planning_cases,
    planning_cases_sha256,
)


def test_planning_evaluator_reports_routing_and_plan_integrity() -> None:
    cases = [
        PlanningBenchmarkCase(
            id="fast",
            goal="Fix the calculator bug",
            expected_task_type=TaskType.BUGFIX,
            expected_complexity=TaskComplexity.SIMPLE,
            expected_risk=RiskLevel.LOW,
            expected_route=RouteKind.FAST,
        ),
        PlanningBenchmarkCase(
            id="guarded",
            goal="Refactor authentication across multiple modules in production",
            expected_task_type=TaskType.REFACTOR,
            expected_complexity=TaskComplexity.COMPLEX,
            expected_risk=RiskLevel.CRITICAL,
            expected_route=RouteKind.HUMAN_GATE,
        ),
    ]

    report = PlanningEvaluator().evaluate(cases)

    assert report.cases == 2
    assert report.route_accuracy == 1.0
    assert report.planning_decision_accuracy == 1.0
    assert report.plan_structural_validity_rate == 1.0
    assert report.failures == []
    assert report.details[0].plan_steps == ["scout", "code", "verify", "review"]
    assert report.details[1].plan_steps == [
        "scout",
        "plan",
        "implementation",
        "test_design",
        "integration",
        "code",
        "verify",
        "review",
    ]


def test_planning_evaluator_surfaces_label_mismatches() -> None:
    case = PlanningBenchmarkCase(
        id="mismatch",
        goal="Fix a small typo",
        expected_task_type=TaskType.FEATURE,
        expected_complexity=TaskComplexity.COMPLEX,
        expected_risk=RiskLevel.HIGH,
        expected_route=RouteKind.COMPLEX,
    )

    report = PlanningEvaluator().evaluate([case])

    assert report.route_accuracy == 0.0
    assert report.failures[0].errors == [
        "task_type_correct",
        "complexity_correct",
        "risk_correct",
        "route_correct",
        "planning_decision_correct",
    ]
    assert report.plan_structural_validity_rate == 1.0


def test_planning_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "planning.json"
    case = {
        "id": "duplicate",
        "goal": "Fix a known defect",
        "expected_task_type": "bugfix",
        "expected_complexity": "simple",
        "expected_risk": "low",
        "expected_route": "fast",
    }
    dataset.write_text(json.dumps([case, case]), encoding="utf-8")

    with pytest.raises(ValueError, match="ids must be unique"):
        load_planning_cases(dataset)


def test_frozen_planning_benchmark_stays_above_resume_claims() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = load_planning_cases(root / "benchmarks" / "planning_cases.json")

    report = PlanningEvaluator().evaluate(cases, split="test")

    assert len(cases) >= 30
    assert report.dataset_sha256 == planning_cases_sha256(cases)
    assert report.dataset_sha256 == (
        "de239c36c17f47c6086d6b81f260fbd7f34987b39a510295989ab8de1f60c3a3"
    )
    assert report.route_accuracy >= 0.85
    assert report.planning_decision_accuracy >= 0.85
    assert report.plan_structural_validity_rate >= 0.99
