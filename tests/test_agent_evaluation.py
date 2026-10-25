import json
import subprocess
from pathlib import Path

import pytest

from ai_harness.agent_evaluation import (
    AgentBenchmarkCase,
    AgentBenchmarkEvaluator,
    AgentBenchmarkObservation,
    aggregate_agent_reports,
    changed_path_metrics,
    dataset_sha256,
    load_agent_cases,
    render_agent_aggregate_markdown,
)
from ai_harness.cli import build_parser
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
        edit_strategies={"exact_anchor": 1},
        edit_failure_codes=(
            {"anchor_not_unique": 1} if status == TaskStatus.FAILED else {}
        ),
    )


def test_paired_evaluation_alternates_order_and_reports_delta():
    cases = [
        AgentBenchmarkCase(id="case-a", goal="Fix the first calculator defect"),
        AgentBenchmarkCase(id="case-b", goal="Fix the second calculator defect"),
    ]
    calls = []

    def runner(case, variant, run_index):
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
    assert report.repetitions == 1
    assert report.observations_per_variant == 2
    assert report.multi.success_rate == 1.0
    assert report.multi_minus_single.success_rate == 0.5
    assert report.multi_minus_single.mean_model_calls == 1.0
    assert report.multi_minus_single.mean_total_tokens == 15.0
    assert report.pairwise_success.multi_wins == 1
    assert report.pairwise_success.ties == 1


def test_repeated_evaluation_counterbalances_variant_order():
    case = AgentBenchmarkCase(id="case-a", goal="Fix the repeated calculator defect")
    calls = []

    def runner(current_case, variant, run_index):
        calls.append((current_case.id, variant))
        return observation(
            current_case.id,
            variant,
            status=TaskStatus.SUCCEEDED,
            duration_ms=10,
            model_calls=1,
            total_tokens=10,
        )

    report = AgentBenchmarkEvaluator().evaluate(
        [case], runner, dataset_hash="frozen-dataset", repetitions=2
    )

    assert calls == [
        ("case-a", "single"),
        ("case-a", "multi"),
        ("case-a", "multi"),
        ("case-a", "single"),
    ]
    assert report.repetitions == 2
    assert report.observations_per_variant == 2
    assert {item.run_index for item in report.details} == {1, 2}


def test_aggregate_reports_separates_provider_and_agent_failures():
    cases = [AgentBenchmarkCase(id="case-a", goal="Fix the repeated calculator defect")]

    def first_runner(case, variant, run_index):
        item = observation(
            case.id,
            variant,
            status=TaskStatus.SUCCEEDED,
            duration_ms=100 if variant == "single" else 200,
            model_calls=1 if variant == "single" else 2,
            total_tokens=20 if variant == "single" else 40,
        )
        if variant == "multi":
            item.status = TaskStatus.FAILED
            item.error = "ReadTimeout: timed out"
            item.total_tokens = None
        return item

    def second_runner(case, variant, run_index):
        item = observation(
            case.id,
            variant,
            status=TaskStatus.SUCCEEDED,
            duration_ms=300 if variant == "single" else 400,
            model_calls=1 if variant == "single" else 2,
            total_tokens=30 if variant == "single" else 50,
        )
        if variant == "single":
            item.status = TaskStatus.FAILED
            item.error = "The proposed edit batch could not be applied safely"
            item.edit_failure_codes = {"anchor_not_unique": 1}
        return item

    evaluator = AgentBenchmarkEvaluator()
    first = evaluator.evaluate(cases, first_runner, dataset_hash="frozen-dataset")
    second = evaluator.evaluate(cases, second_runner, dataset_hash="frozen-dataset")

    report = aggregate_agent_reports([first, second])

    assert report.runs == 2
    assert report.observations_per_variant == 2
    assert report.single.raw_success_rate == 0.5
    assert report.single.provider_failures == 0
    assert report.single.agent_failures == 1
    assert report.single.failure_classes == {"edit_application": 1}
    assert report.single.edit_strategies == {"exact_anchor": 2}
    assert report.single.edit_failure_codes == {"anchor_not_unique": 1}
    assert report.multi.raw_success_rate == 0.5
    assert report.multi.provider_failures == 1
    assert report.multi.agent_failures == 0
    assert report.multi.provider_adjusted_success_rate == 1.0
    assert report.multi.known_token_observations == 1
    assert report.multi.mean_known_tokens == 50.0
    assert [run.run_index for run in report.per_run] == [1, 2]
    markdown = render_agent_aggregate_markdown(report)
    assert "Provider-adjusted success" in markdown
    assert "`provider_failure`: 1" in markdown
    assert "## Edit application" in markdown
    assert "`anchor_not_unique`: 1" in markdown
    assert "## Resume-safe result" in markdown
    assert "single-agent execution achieved 50.00% end-to-end success" in markdown


def test_aggregate_reports_rejects_mismatched_datasets():
    case = AgentBenchmarkCase(id="case-a", goal="Fix the repeated calculator defect")

    def runner(current_case, variant, run_index):
        return observation(
            current_case.id,
            variant,
            status=TaskStatus.SUCCEEDED,
            duration_ms=10,
            model_calls=1,
            total_tokens=10,
        )

    evaluator = AgentBenchmarkEvaluator()
    first = evaluator.evaluate([case], runner, dataset_hash="first")
    second = evaluator.evaluate([case], runner, dataset_hash="second")

    with pytest.raises(ValueError, match="different datasets"):
        aggregate_agent_reports([first, second])


def test_trusted_fixture_approval_is_an_explicit_benchmark_opt_in():
    parser = build_parser()

    default_args = parser.parse_args(["eval-agents", "repo", "cases.json"])
    approved_args = parser.parse_args(
        [
            "eval-agents",
            "repo",
            "cases.json",
            "--approve-trusted-fixtures",
        ]
    )

    assert default_args.approve_trusted_fixtures is False
    assert approved_args.approve_trusted_fixtures is True


def test_agent_report_summary_cli_accepts_json_and_markdown_outputs():
    args = build_parser().parse_args(
        [
            "summarize-agent-reports",
            "run-one.json",
            "run-two.json",
            "--output",
            "aggregate.json",
            "--markdown-output",
            "aggregate.md",
        ]
    )

    assert args.reports == ["run-one.json", "run-two.json"]
    assert args.output == "aggregate.json"
    assert args.markdown_output == "aggregate.md"


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


def test_frozen_agent_dataset_has_thirty_independently_failing_tasks():
    root = Path(__file__).resolve().parents[1]
    fixture_root = root / "benchmarks" / "fixtures" / "agent_eval"
    dataset = root / "benchmarks" / "agent_cases.json"
    cases = load_agent_cases(dataset)

    assert len(cases) == 30
    assert dataset_sha256(dataset) == (
        "6f9d43944a941efd78d2ae6534f62eed5a4a5e794820687c5c79ae1b6c72e167"
    )
    for case in cases:
        assert case.expected_changed_paths
        assert all((fixture_root / path).is_file() for path in case.expected_changed_paths)
        for command in case.verification_commands:
            completed = subprocess.run(
                command.split(),
                cwd=fixture_root,
                capture_output=True,
                check=False,
                text=True,
            )
            assert completed.returncode != 0, f"{case.id} is not a failing fixture"
