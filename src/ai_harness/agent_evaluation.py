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
        single = self._summarize(single_items)
        multi = self._summarize(multi_items)
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

    @staticmethod
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
            mean_repair_rounds=round(
                sum(item.repair_rounds for item in items) / count, 3
            ),
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
