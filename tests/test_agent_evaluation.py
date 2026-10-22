import json
from pathlib import Path

import pytest

from ai_harness.agent_evaluation import (
    AgentBenchmarkCase,
    AgentBenchmarkEvaluator,
    AgentBenchmarkObservation,
    changed_path_metrics,
    dataset_sha256,
    load_agent_cases,
)
from ai_harness.models import TaskStatus


def observation(
    case_id: str,
    variant: str,
    *,
    status: TaskStatus,
    duration_ms: int,
    model_calls: int,
    total_tokens: int,
) -> AgentBenchmarkObservation:
    return AgentBenchmarkObservation(
        case_id=case_id,
        variant=variant,
        status=status,
        duration_ms=duration_ms,
        model_calls=model_calls,
        request_chars=model_calls * 100,
        prompt_tokens=total_tokens - 5,
        completion_tokens=5,
        total_tokens=total_tokens,
        repair_rounds=0,
        collaboration_rounds=1 if variant == "multi" else 0,
        detected_conflicts=0,
        changed_paths=["calculator.py"],
        changed_path_recall=1.0,
        changed_path_precision=1.0,
    )


def test_paired_evaluation_alternates_order_and_reports_delta():
    cases = [
        AgentBenchmarkCase(id="case-a", goal="Fix the first calculator defect"),
        AgentBenchmarkCase(id="case-b", goal="Fix the second calculator defect"),
    ]
    calls = []

    def runner(case, variant):
        calls.append((case.id, variant))
        if case.id == "case-a" and variant == "single":
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.SUCCEEDED
        return observation(
            case.id,
            variant,
            status=status,
            duration_ms=100 if variant == "single" else 150,
            model_calls=1 if variant == "single" else 2,
            total_tokens=20 if variant == "single" else 35,
        )

    report = AgentBenchmarkEvaluator().evaluate(
        cases,
        runner,
        dataset_hash="frozen-dataset",
    )

    assert calls == [
        ("case-a", "single"),
        ("case-a", "multi"),
        ("case-b", "multi"),
        ("case-b", "single"),
    ]
    assert report.single.success_rate == 0.5
    assert report.multi.success_rate == 1.0
    assert report.multi_minus_single.success_rate == 0.5
    assert report.multi_minus_single.mean_model_calls == 1.0
    assert report.multi_minus_single.mean_total_tokens == 15.0
    assert report.pairwise_success.multi_wins == 1
    assert report.pairwise_success.ties == 1


def test_dataset_loading_hash_and_changed_path_metrics(tmp_path: Path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "calculator-add",
                    "goal": "Fix calculator addition behavior",
                    "expected_changed_paths": ["calculator.py", "tests/test_calculator.py"],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_agent_cases(dataset)
    recall, precision = changed_path_metrics(
        ["calculator.py", "README.md"],
        cases[0].expected_changed_paths,
    )

    assert cases[0].id == "calculator-add"
    assert len(dataset_sha256(dataset)) == 64
    assert recall == 0.5
    assert precision == 0.5


def test_dataset_must_be_an_array(tmp_path: Path):
    dataset = tmp_path / "cases.json"
    dataset.write_text('{"id": "not-an-array"}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        load_agent_cases(dataset)


def test_dataset_case_ids_must_be_unique(tmp_path: Path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            [
                {"id": "duplicate", "goal": "Fix the first known defect"},
                {"id": "duplicate", "goal": "Fix the second known defect"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be unique"):
        load_agent_cases(dataset)
