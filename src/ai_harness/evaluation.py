from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from .repository import ContextIndex, RepoMapBuilder, Repository, RetrievalCache


class RetrievalCase(BaseModel):
    id: str
    query: str = Field(min_length=3)
    relevant_paths: List[str] = Field(min_length=1)


class CaseMetrics(BaseModel):
    id: str
    retrieved_paths: List[str]
    relevant_paths: List[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    hit: bool
    context_reduction_ratio: float


class RetrievalEvaluation(BaseModel):
    cases: int
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate: float
    mean_context_reduction_ratio: float
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

    def evaluate(self, cases: List[RetrievalCase], k: int = 5) -> RetrievalEvaluation:
        if not cases:
            raise ValueError("at least one retrieval case is required")
        if k <= 0:
            raise ValueError("k must be positive")
        details: List[CaseMetrics] = []
        for case in cases:
            index = ContextIndex(self.repository, self.repo_map, cache=self.cache)
            hits = index.search(case.query, limit=k)
            retrieved = list(dict.fromkeys(hit.path for hit in hits))[:k]
            relevant = set(case.relevant_paths)
            matched = relevant.intersection(retrieved)
            first_rank = next(
                (rank for rank, path in enumerate(retrieved, start=1) if path in relevant),
                None,
            )
            details.append(
                CaseMetrics(
                    id=case.id,
                    retrieved_paths=retrieved,
                    relevant_paths=case.relevant_paths,
                    recall_at_k=round(len(matched) / len(relevant), 6),
                    precision_at_k=round(len(matched) / k, 6),
                    reciprocal_rank=round(1 / first_rank, 6) if first_rank else 0.0,
                    hit=bool(matched),
                    context_reduction_ratio=index.last_trace.context_reduction_ratio,
                )
            )
        count = len(details)
        return RetrievalEvaluation(
            cases=count,
            k=k,
            recall_at_k=round(sum(item.recall_at_k for item in details) / count, 6),
            precision_at_k=round(sum(item.precision_at_k for item in details) / count, 6),
            mrr=round(sum(item.reciprocal_rank for item in details) / count, 6),
            hit_rate=round(sum(item.hit for item in details) / count, 6),
            mean_context_reduction_ratio=round(
                sum(item.context_reduction_ratio for item in details) / count, 6
            ),
            details=details,
        )
