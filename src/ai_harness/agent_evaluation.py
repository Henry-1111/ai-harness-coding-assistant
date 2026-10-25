from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import TaskStatus, utc_now


class AgentBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    goal: str = Field(min_length=5, max_length=4000)
    acceptance_criteria: List[str] = Field(default_factory=list)
    file_scope: List[str] = Field(default_factory=list)
    verification_commands: List[str] = Field(
        default_factory=lambda: ["pytest -q"], min_length=1
    )
    expected_changed_paths: List[str] = Field(default_factory=list)
    max_replans: int = Field(default=1, ge=0, le=3)
    split: str = "test"


class AgentBenchmarkObservation(BaseModel):
    case_id: str
    variant: Literal["single", "multi"]
    run_index: int = Field(default=1, ge=1)
    status: TaskStatus
    duration_ms: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    request_chars: int = Field(ge=0)
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    repair_rounds: int = Field(ge=0)
    collaboration_rounds: int = Field(ge=0)
    detected_conflicts: int = Field(ge=0)
    changed_paths: List[str] = Field(default_factory=list)
    changed_path_recall: Optional[float] = Field(default=None, ge=0, le=1)
    changed_path_precision: Optional[float] = Field(default=None, ge=0, le=1)
    error: Optional[str] = None


class AgentVariantSummary(BaseModel):
    cases: int
    success_rate: float
    mean_duration_ms: float
    mean_model_calls: float
    mean_request_chars: float
    mean_total_tokens: Optional[float]
    mean_repair_rounds: float
    mean_collaboration_rounds: float
    mean_detected_conflicts: float
    mean_changed_path_recall: Optional[float]
    mean_changed_path_precision: Optional[float]


class AgentPairwise(BaseModel):
    multi_wins: int = 0
    ties: int = 0
    single_wins: int = 0


class AgentBenchmarkDelta(BaseModel):
    success_rate: float
    mean_duration_ms: float
    mean_model_calls: float
    mean_total_tokens: Optional[float]
    mean_repair_rounds: float


class AgentBenchmarkReport(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    dataset_sha256: str
    cases: int
    repetitions: int
    observations_per_variant: int
    single: AgentVariantSummary
    multi: AgentVariantSummary
    multi_minus_single: AgentBenchmarkDelta
    pairwise_success: AgentPairwise
    details: List[AgentBenchmarkObservation]


class AgentAggregateRun(BaseModel):
    run_index: int
    single_success_rate: float
    multi_success_rate: float
    single_provider_failures: int
    multi_provider_failures: int


class AgentAggregateVariant(BaseModel):
    observations: int
    successes: int
    raw_success_rate: float
    provider_failures: int
    agent_failures: int
    provider_adjusted_observations: int
    provider_adjusted_success_rate: float
    mean_duration_ms: float
    p50_duration_ms: int
    p95_duration_ms: int
    mean_model_calls: float
    known_token_observations: int
    known_total_tokens: int
    mean_known_tokens: Optional[float]
    mean_changed_path_recall: Optional[float]
    mean_changed_path_precision: Optional[float]
    failure_classes: Dict[str, int]


class AgentAggregateReport(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    dataset_sha256: str
    cases: int
    runs: int
    observations_per_variant: int
    single: AgentAggregateVariant
    multi: AgentAggregateVariant
    per_run: List[AgentAggregateRun]


def load_agent_cases(path: Path) -> List[AgentBenchmarkCase]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("agent benchmark dataset must be a JSON array")
    cases = [AgentBenchmarkCase.model_validate(item) for item in value]
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("agent benchmark case ids must be unique")
    return cases


def dataset_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class AgentBenchmarkEvaluator:
    """Run paired single/multi-agent cases and report cost-quality trade-offs."""

    def evaluate(
        self,
        cases: List[AgentBenchmarkCase],
        runner: Callable[
            [AgentBenchmarkCase, Literal["single", "multi"], int],
            AgentBenchmarkObservation,
        ],
        *,
        dataset_hash: str,
        split: Optional[str] = None,
        repetitions: int = 1,
    ) -> AgentBenchmarkReport:
        selected = [case for case in cases if not split or case.split == split]
        if not selected:
            raise ValueError("at least one agent benchmark case is required")
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        details: List[AgentBenchmarkObservation] = []
        paired: Dict[tuple[str, int], Dict[str, AgentBenchmarkObservation]] = {}
        for repetition in range(repetitions):
            for index, case in enumerate(selected):
                variants = (
                    ("single", "multi")
                    if (index + repetition) % 2 == 0
                    else ("multi", "single")
                )
                pair_key = (case.id, repetition + 1)
                paired[pair_key] = {}
                for variant in variants:
                    observation = runner(case, variant, repetition + 1).model_copy(
                        update={"run_index": repetition + 1}
                    )
                    if observation.case_id != case.id or observation.variant != variant:
                        raise ValueError("benchmark runner returned mismatched case or variant")
                    details.append(observation)
                    paired[pair_key][variant] = observation
        single_items = [pair["single"] for pair in paired.values()]
        multi_items = [pair["multi"] for pair in paired.values()]
        single = _summarize(single_items)
        multi = _summarize(multi_items)
        pairwise = AgentPairwise()
        for pair in paired.values():
            single_success = pair["single"].status == TaskStatus.SUCCEEDED
            multi_success = pair["multi"].status == TaskStatus.SUCCEEDED
            if multi_success and not single_success:
                pairwise.multi_wins += 1
            elif single_success and not multi_success:
                pairwise.single_wins += 1
            else:
                pairwise.ties += 1
        return AgentBenchmarkReport(
            dataset_sha256=dataset_hash,
            cases=len(selected),
            repetitions=repetitions,
            observations_per_variant=len(selected) * repetitions,
            single=single,
            multi=multi,
            multi_minus_single=AgentBenchmarkDelta(
                success_rate=round(multi.success_rate - single.success_rate, 6),
                mean_duration_ms=round(multi.mean_duration_ms - single.mean_duration_ms, 3),
                mean_model_calls=round(multi.mean_model_calls - single.mean_model_calls, 3),
                mean_total_tokens=(
                    round(multi.mean_total_tokens - single.mean_total_tokens, 3)
                    if multi.mean_total_tokens is not None
                    and single.mean_total_tokens is not None
                    else None
                ),
                mean_repair_rounds=round(
                    multi.mean_repair_rounds - single.mean_repair_rounds, 3
                ),
            ),
            pairwise_success=pairwise,
            details=details,
        )


def _summarize(items: List[AgentBenchmarkObservation]) -> AgentVariantSummary:
    count = len(items)
    token_values = [item.total_tokens for item in items]
    recall_values = [
        item.changed_path_recall
        for item in items
        if item.changed_path_recall is not None
    ]
    precision_values = [
        item.changed_path_precision
        for item in items
        if item.changed_path_precision is not None
    ]
    return AgentVariantSummary(
        cases=count,
        success_rate=round(
            sum(item.status == TaskStatus.SUCCEEDED for item in items) / count, 6
        ),
        mean_duration_ms=round(sum(item.duration_ms for item in items) / count, 3),
        mean_model_calls=round(sum(item.model_calls for item in items) / count, 3),
        mean_request_chars=round(sum(item.request_chars for item in items) / count, 3),
        mean_total_tokens=(
            round(sum(value for value in token_values if value is not None) / count, 3)
            if all(value is not None for value in token_values)
            else None
        ),
        mean_repair_rounds=round(sum(item.repair_rounds for item in items) / count, 3),
        mean_collaboration_rounds=round(
            sum(item.collaboration_rounds for item in items) / count, 3
        ),
        mean_detected_conflicts=round(
            sum(item.detected_conflicts for item in items) / count, 3
        ),
        mean_changed_path_recall=(
            round(sum(recall_values) / len(recall_values), 6) if recall_values else None
        ),
        mean_changed_path_precision=(
            round(sum(precision_values) / len(precision_values), 6)
            if precision_values
            else None
        ),
    )


def _is_provider_failure(observation: AgentBenchmarkObservation) -> bool:
    error = observation.error or ""
    markers = (
        "ReadTimeout",
        "ConnectError",
        "ConnectionError",
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
        "upstream_timeout",
        "upstream_connection_error",
    )
    return any(marker in error for marker in markers)


def _failure_class(observation: AgentBenchmarkObservation) -> str:
    if observation.status == TaskStatus.SUCCEEDED:
        return "succeeded"
    if _is_provider_failure(observation):
        return "provider_failure"
    error = observation.error or ""
    if "edit batch" in error or "old_text" in error:
        return "edit_application"
    if "repeated" in error:
        return "repeated_verification"
    if "Verification failed" in error:
        return "verification"
    if observation.status == TaskStatus.NEEDS_INPUT or "approval" in error.lower():
        return "approval"
    return "other_agent_failure"


def _percentile(values: List[int], percentile: float) -> int:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * percentile)
    return ordered[index]


def _aggregate_variant(
    items: List[AgentBenchmarkObservation],
) -> AgentAggregateVariant:
    count = len(items)
    successes = sum(item.status == TaskStatus.SUCCEEDED for item in items)
    provider_failures = sum(_is_provider_failure(item) for item in items)
    valid_count = count - provider_failures
    token_values = [item.total_tokens for item in items if item.total_tokens is not None]
    recall_values = [
        item.changed_path_recall
        for item in items
        if item.changed_path_recall is not None
    ]
    precision_values = [
        item.changed_path_precision
        for item in items
        if item.changed_path_precision is not None
    ]
    failure_classes: Dict[str, int] = {}
    for item in items:
        failure_class = _failure_class(item)
        if failure_class == "succeeded":
            continue
        failure_classes[failure_class] = failure_classes.get(failure_class, 0) + 1
    return AgentAggregateVariant(
        observations=count,
        successes=successes,
        raw_success_rate=round(successes / count, 6),
        provider_failures=provider_failures,
        agent_failures=count - successes - provider_failures,
        provider_adjusted_observations=valid_count,
        provider_adjusted_success_rate=(
            round(successes / valid_count, 6) if valid_count else 0.0
        ),
        mean_duration_ms=round(sum(item.duration_ms for item in items) / count, 3),
        p50_duration_ms=_percentile([item.duration_ms for item in items], 0.5),
        p95_duration_ms=_percentile([item.duration_ms for item in items], 0.95),
        mean_model_calls=round(sum(item.model_calls for item in items) / count, 3),
        known_token_observations=len(token_values),
        known_total_tokens=sum(token_values),
        mean_known_tokens=(
            round(sum(token_values) / len(token_values), 3) if token_values else None
        ),
        mean_changed_path_recall=(
            round(sum(recall_values) / len(recall_values), 6) if recall_values else None
        ),
        mean_changed_path_precision=(
            round(sum(precision_values) / len(precision_values), 6)
            if precision_values
            else None
        ),
        failure_classes=dict(sorted(failure_classes.items())),
    )


def aggregate_agent_reports(
    reports: List[AgentBenchmarkReport],
) -> AgentAggregateReport:
    if not reports:
        raise ValueError("at least one agent benchmark report is required")
    dataset_hash = reports[0].dataset_sha256
    cases = reports[0].cases
    if any(report.dataset_sha256 != dataset_hash for report in reports):
        raise ValueError("agent benchmark reports use different datasets")
    if any(report.cases != cases for report in reports):
        raise ValueError("agent benchmark reports use different case counts")

    all_items: List[AgentBenchmarkObservation] = []
    per_run: List[AgentAggregateRun] = []
    global_run = 0
    for report in reports:
        for local_run in range(1, report.repetitions + 1):
            global_run += 1
            run_items = [item for item in report.details if item.run_index == local_run]
            single_items = [item for item in run_items if item.variant == "single"]
            multi_items = [item for item in run_items if item.variant == "multi"]
            if len(single_items) != cases or len(multi_items) != cases:
                raise ValueError("agent benchmark report has incomplete paired observations")
            all_items.extend(run_items)
            per_run.append(
                AgentAggregateRun(
                    run_index=global_run,
                    single_success_rate=round(
                        sum(item.status == TaskStatus.SUCCEEDED for item in single_items)
                        / cases,
                        6,
                    ),
                    multi_success_rate=round(
                        sum(item.status == TaskStatus.SUCCEEDED for item in multi_items)
                        / cases,
                        6,
                    ),
                    single_provider_failures=sum(
                        _is_provider_failure(item) for item in single_items
                    ),
                    multi_provider_failures=sum(
                        _is_provider_failure(item) for item in multi_items
                    ),
                )
            )

    single = _aggregate_variant([item for item in all_items if item.variant == "single"])
    multi = _aggregate_variant([item for item in all_items if item.variant == "multi"])
    return AgentAggregateReport(
        dataset_sha256=dataset_hash,
        cases=cases,
        runs=global_run,
        observations_per_variant=cases * global_run,
        single=single,
        multi=multi,
        per_run=per_run,
    )


def render_agent_aggregate_markdown(report: AgentAggregateReport) -> str:
    def percent(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        "# Agent benchmark aggregate",
        "",
        f"Dataset SHA-256: `{report.dataset_sha256}`  ",
        f"Cases: {report.cases}  ",
        f"Runs: {report.runs}  ",
        f"Observations per variant: {report.observations_per_variant}",
        "",
        "| Metric | Single agent | Multi agent |",
        "| --- | ---: | ---: |",
        (
            f"| Raw success | {report.single.successes}/{report.single.observations} "
            f"({percent(report.single.raw_success_rate)}) | "
            f"{report.multi.successes}/{report.multi.observations} "
            f"({percent(report.multi.raw_success_rate)}) |"
        ),
        (
            "| Provider-adjusted success | "
            f"{percent(report.single.provider_adjusted_success_rate)} | "
            f"{percent(report.multi.provider_adjusted_success_rate)} |"
        ),
        (
            f"| P50 duration | {report.single.p50_duration_ms:,} ms | "
            f"{report.multi.p50_duration_ms:,} ms |"
        ),
        (
            f"| P95 duration | {report.single.p95_duration_ms:,} ms | "
            f"{report.multi.p95_duration_ms:,} ms |"
        ),
        (
            f"| Mean model calls | {report.single.mean_model_calls:.3f} | "
            f"{report.multi.mean_model_calls:.3f} |"
        ),
        (
            "| Mean known tokens | "
            f"{(report.single.mean_known_tokens or 0):,.0f} | "
            f"{(report.multi.mean_known_tokens or 0):,.0f} |"
        ),
        (
            f"| Provider failures | {report.single.provider_failures} | "
            f"{report.multi.provider_failures} |"
        ),
        f"| Agent failures | {report.single.agent_failures} | {report.multi.agent_failures} |",
        "",
        "## Per-run raw success",
        "",
        "| Run | Single agent | Multi agent | Single Provider failures | Multi Provider failures |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.per_run:
        lines.append(
            f"| {run.run_index} | {percent(run.single_success_rate)} | "
            f"{percent(run.multi_success_rate)} | {run.single_provider_failures} | "
            f"{run.multi_provider_failures} |"
        )
    lines.extend(["", "## Failure classes", ""])
    for variant, summary in (("Single agent", report.single), ("Multi agent", report.multi)):
        lines.append(f"### {variant}")
        lines.append("")
        if summary.failure_classes:
            for name, count in summary.failure_classes.items():
                lines.append(f"- `{name}`: {count}")
        else:
            lines.append("- None")
        lines.append("")
    lines.append(
        "Provider-adjusted success is diagnostic only; raw success remains the end-to-end metric."
    )
    lines.extend(
        [
            "",
            "## Resume-safe result",
            "",
            (
                f"On the frozen {report.cases}-case dataset across {report.runs} runs, "
                f"single-agent execution achieved {percent(report.single.raw_success_rate)} "
                f"end-to-end success with P95 latency of "
                f"{report.single.p95_duration_ms:,} ms and mean observed usage of "
                f"{(report.single.mean_known_tokens or 0):,.0f} tokens."
            ),
            "",
            (
                "The Provider-adjusted diagnostic success rate was "
                f"{percent(report.single.provider_adjusted_success_rate)}; this excludes "
                f"{report.single.provider_failures} explicit Provider failures and must not "
                "replace the end-to-end rate."
            ),
        ]
    )
    return "\n".join(lines) + "\n"

def changed_path_metrics(
    changed_paths: List[str], expected_paths: List[str]
) -> tuple[Optional[float], Optional[float]]:
    if not expected_paths:
        return None, None
    changed = set(changed_paths)
    expected = set(expected_paths)
    matched = changed.intersection(expected)
    return (
        round(len(matched) / len(expected), 6),
        round(len(matched) / max(len(changed), 1), 6),
    )
