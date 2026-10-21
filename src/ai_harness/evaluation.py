from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .repository import (
    ContextIndex,
    ProgressiveRetriever,
    RepoMapBuilder,
    Repository,
    RetrievalCache,
)


class RetrievalCase(BaseModel):
    id: str
    query: str = Field(min_length=3)
    relevant_paths: List[str] = Field(min_length=1)
    language: str = "unknown"
    task_type: str = "general"
    split: str = "test"


class CaseMetrics(BaseModel):
    id: str
    retrieved_paths: List[str]
    relevant_paths: List[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    hit: bool
    context_reduction_ratio: float
    rounds: int
    stop_reason: str
    baseline_recall_at_k: float


class EvaluationSlice(BaseModel):
    cases: int
    recall_at_k: float
    mrr: float
    hit_rate: float


class RetrievalEvaluation(BaseModel):
    cases: int
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    mean_context_reduction_ratio: float
    baseline_recall_at_k: float
    recall_delta: float
    by_language: Dict[str, EvaluationSlice] = Field(default_factory=dict)
    by_task_type: Dict[str, EvaluationSlice] = Field(default_factory=dict)
    details: List[CaseMetrics]


def load_cases(path: Path) -> List[RetrievalCase]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("retrieval dataset must be a JSON array")
    return [RetrievalCase.model_validate(item) for item in value]


class RetrievalEvaluator:
    """Evaluate file-level repository retrieval against a fixed labelled task set."""

    def __init__(self, repository: Repository) -> None:
        self.repository = repository
        self.repo_map = RepoMapBuilder().build(repository)
        self.cache = RetrievalCache()

    def evaluate(
        self, cases: List[RetrievalCase], k: int = 5, split: Optional[str] = None
    ) -> RetrievalEvaluation:
        if split:
            cases = [case for case in cases if case.split == split]
        if not cases:
            raise ValueError("at least one retrieval case is required")
        if k <= 0:
            raise ValueError("k must be positive")
        details: List[CaseMetrics] = []
        for case in cases:
            baseline_index = ContextIndex(self.repository, self.repo_map, cache=self.cache)
            baseline_hits = baseline_index.search(case.query, limit=k)
            baseline_retrieved = list(dict.fromkeys(hit.path for hit in baseline_hits))[:k]
            index = ProgressiveRetriever(self.repository, self.repo_map, cache=self.cache)
            hits = index.search(case.query, limit=k)
            retrieved = list(dict.fromkeys(hit.path for hit in hits))[:k]
            relevant = set(case.relevant_paths)
            matched = relevant.intersection(retrieved)
            first_rank = next(
                (rank for rank, path in enumerate(retrieved, start=1) if path in relevant),
                None,
            )
            dcg = sum(
                1 / math.log2(rank + 1)
                for rank, path in enumerate(retrieved, start=1)
                if path in relevant
            )
            ideal_dcg = sum(
                1 / math.log2(rank + 1)
                for rank in range(1, min(len(relevant), k) + 1)
            )
            details.append(
                CaseMetrics(
                    id=case.id,
                    retrieved_paths=retrieved,
                    relevant_paths=case.relevant_paths,
                    recall_at_k=round(len(matched) / len(relevant), 6),
                    precision_at_k=round(len(matched) / k, 6),
                    reciprocal_rank=round(1 / first_rank, 6) if first_rank else 0.0,
                    ndcg_at_k=round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0,
                    hit=bool(matched),
                    context_reduction_ratio=index.last_trace.context_reduction_ratio,
                    rounds=index.last_trace.rounds,
                    stop_reason=index.last_trace.stop_reason,
                    baseline_recall_at_k=round(
                        len(relevant.intersection(baseline_retrieved)) / len(relevant), 6
                    ),
                )
            )
        count = len(details)
        by_language = self._group(cases, details, "language")
        by_task_type = self._group(cases, details, "task_type")
        recall = round(sum(item.recall_at_k for item in details) / count, 6)
        baseline_recall = round(
            sum(item.baseline_recall_at_k for item in details) / count, 6
        )
        return RetrievalEvaluation(
            cases=count,
            k=k,
            recall_at_k=recall,
            precision_at_k=round(sum(item.precision_at_k for item in details) / count, 6),
            mrr=round(sum(item.reciprocal_rank for item in details) / count, 6),
            ndcg_at_k=round(sum(item.ndcg_at_k for item in details) / count, 6),
            hit_rate=round(sum(item.hit for item in details) / count, 6),
            mean_context_reduction_ratio=round(
                sum(item.context_reduction_ratio for item in details) / count, 6
            ),
            baseline_recall_at_k=baseline_recall,
            recall_delta=round(recall - baseline_recall, 6),
            by_language=by_language,
            by_task_type=by_task_type,
            details=details,
        )

    @staticmethod
    def _group(
        cases: List[RetrievalCase], details: List[CaseMetrics], field: str
    ) -> Dict[str, EvaluationSlice]:
        case_by_id = {case.id: case for case in cases}
        groups: Dict[str, List[CaseMetrics]] = defaultdict(list)
        for detail in details:
            groups[str(getattr(case_by_id[detail.id], field))].append(detail)
        return {
            name: EvaluationSlice(
                cases=len(items),
                recall_at_k=round(sum(item.recall_at_k for item in items) / len(items), 6),
                mrr=round(sum(item.reciprocal_rank for item in items) / len(items), 6),
                hit_rate=round(sum(item.hit for item in items) / len(items), 6),
            )
            for name, items in sorted(groups.items())
        }
