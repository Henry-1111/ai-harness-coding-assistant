from __future__ import annotations

import json
import math
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import RepoFile
from .repository import (
    ContextIndex,
    ProgressiveRetriever,
    RepoMapBuilder,
    Repository,
    RetrievalCache,
)


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    query: str = Field(min_length=3)
    relevant_paths: List[str] = Field(min_length=1)
    language: str = "unknown"
    task_type: str = "general"
    split: Literal["train", "validation", "test"] = "test"


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
    baseline_reciprocal_rank: float
    stage_relevant_ranks: Dict[str, Dict[str, Optional[int]]] = Field(default_factory=dict)


class BadCase(BaseModel):
    id: str
    reasons: List[str]
    missing_paths: List[str]
    retrieved_paths: List[str]


class PairwiseComparison(BaseModel):
    metric: str = "reciprocal_rank"
    progressive_wins: int = 0
    ties: int = 0
    baseline_wins: int = 0


class EvaluationSlice(BaseModel):
    cases: int
    recall_at_k: float
    mrr: float
    hit_rate: float


class RetrievalEvaluation(BaseModel):
    dataset_sha256: str
    split: Optional[str] = None
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
    baseline_mrr: float
    mrr_delta: float
    pairwise: PairwiseComparison
    bad_cases: List[BadCase]
    by_language: Dict[str, EvaluationSlice] = Field(default_factory=dict)
    by_task_type: Dict[str, EvaluationSlice] = Field(default_factory=dict)
    details: List[CaseMetrics]


class StructureComparison(BaseModel):
    dataset_sha256: str
    repository_sha256: str
    cases: int
    k: int
    structured: RetrievalEvaluation
    without_symbols: RetrievalEvaluation
    recall_delta: float
    mrr_delta: float
    structured_wins: int
    ties: int
    without_symbols_wins: int


def load_cases(
    path: Path, *, repository_root: Optional[Path] = None
) -> List[RetrievalCase]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("cases"), list):
        value = value["cases"]
    elif not isinstance(value, list):
        raise ValueError("retrieval dataset must be a JSON array or an object with cases")
    cases = [RetrievalCase.model_validate(item) for item in value]
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval case ids must be unique")
    queries = [" ".join(case.query.lower().split()) for case in cases]
    if len(queries) != len(set(queries)):
        raise ValueError("retrieval case queries must be unique")
    root = repository_root.resolve() if repository_root is not None else None
    for case in cases:
        if len(case.relevant_paths) != len(set(case.relevant_paths)):
            raise ValueError(f"duplicate relevant path in case {case.id}")
        for raw_path in case.relevant_paths:
            relative = Path(raw_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe relevant path in case {case.id}: {raw_path}")
            if raw_path.lower() in case.query.lower():
                raise ValueError(f"query leaks relevant path in case {case.id}: {raw_path}")
            if root is not None and not (root / relative).is_file():
                raise ValueError(f"relevant path does not exist in case {case.id}: {raw_path}")
    return cases


def retrieval_cases_sha256(cases: List[RetrievalCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class RetrievalEvaluator:
    """Evaluate file-level repository retrieval against a fixed labelled task set."""

    def __init__(
        self, repository: Repository, *, repo_map: Optional[List[RepoFile]] = None
    ) -> None:
        self.repository = repository
        self.repo_map = repo_map if repo_map is not None else RepoMapBuilder().build(repository)
        self.cache = RetrievalCache()

    def evaluate(
        self, cases: List[RetrievalCase], k: int = 5, split: Optional[str] = None
    ) -> RetrievalEvaluation:
        dataset_hash = retrieval_cases_sha256(cases)
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
            baseline_first_rank = next(
                (
                    rank
                    for rank, path in enumerate(baseline_retrieved, start=1)
                    if path in relevant
                ),
                None,
            )
            stage_relevant_ranks = {
                f"{stage.stage}@{stage.round}": {
                    path: (
                        stage.selected_paths.index(path) + 1
                        if path in stage.selected_paths
                        else None
                    )
                    for path in case.relevant_paths
                }
                for stage in index.last_trace.stages
                if stage.stage in {"forge", "focus", "deepen", "related_import", "select"}
            }
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
                    baseline_reciprocal_rank=(
                        round(1 / baseline_first_rank, 6) if baseline_first_rank else 0.0
                    ),
                    stage_relevant_ranks=stage_relevant_ranks,
                )
            )
        count = len(details)
        by_language = self._group(cases, details, "language")
        by_task_type = self._group(cases, details, "task_type")
        recall = round(sum(item.recall_at_k for item in details) / count, 6)
        baseline_recall = round(
            sum(item.baseline_recall_at_k for item in details) / count, 6
        )
        mrr = round(sum(item.reciprocal_rank for item in details) / count, 6)
        baseline_mrr = round(
            sum(item.baseline_reciprocal_rank for item in details) / count, 6
        )
        pairwise = PairwiseComparison()
        for item in details:
            if item.reciprocal_rank > item.baseline_reciprocal_rank:
                pairwise.progressive_wins += 1
            elif item.reciprocal_rank < item.baseline_reciprocal_rank:
                pairwise.baseline_wins += 1
            else:
                pairwise.ties += 1
        bad_cases = self._bad_cases(details)
        return RetrievalEvaluation(
            dataset_sha256=dataset_hash,
            split=split,
            cases=count,
            k=k,
            recall_at_k=recall,
            precision_at_k=round(sum(item.precision_at_k for item in details) / count, 6),
            mrr=mrr,
            ndcg_at_k=round(sum(item.ndcg_at_k for item in details) / count, 6),
            hit_rate=round(sum(item.hit for item in details) / count, 6),
            mean_context_reduction_ratio=round(
                sum(item.context_reduction_ratio for item in details) / count, 6
            ),
            baseline_recall_at_k=baseline_recall,
            recall_delta=round(recall - baseline_recall, 6),
            baseline_mrr=baseline_mrr,
            mrr_delta=round(mrr - baseline_mrr, 6),
            pairwise=pairwise,
            bad_cases=bad_cases,
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

    @staticmethod
    def _bad_cases(details: List[CaseMetrics]) -> List[BadCase]:
        result = []
        for item in details:
            reasons = []
            if item.recall_at_k < 1.0:
                reasons.append("missed_relevant_files")
            if item.reciprocal_rank < 1.0:
                reasons.append("first_relevant_not_ranked_first")
            if item.stop_reason in {"budget_exhausted", "max_rounds", "no_new_evidence"}:
                reasons.append(item.stop_reason)
            if not reasons:
                continue
            result.append(
                BadCase(
                    id=item.id,
                    reasons=reasons,
                    missing_paths=sorted(
                        set(item.relevant_paths) - set(item.retrieved_paths)
                    ),
                    retrieved_paths=item.retrieved_paths,
                )
            )
        return result


def compare_structure_index(
    repository: Repository,
    cases: List[RetrievalCase],
    *,
    k: int = 5,
    split: Optional[str] = None,
) -> StructureComparison:
    """Ablate symbols/imports while retaining paths, full text, and retrieval settings."""
    repo_map = RepoMapBuilder().build(repository)
    repository_digest = sha256()
    for item in sorted(repo_map, key=lambda entry: entry.path):
        repository_digest.update(item.path.encode("utf-8") + b"\0")
        repository_digest.update(repository.resolve(item.path).read_bytes())
        repository_digest.update(b"\0")
    structured = RetrievalEvaluator(repository, repo_map=repo_map).evaluate(
        cases, k=k, split=split
    )
    without_symbols_map = [
        item.model_copy(update={"symbols": [], "imports": []}) for item in repo_map
    ]
    without_symbols = RetrievalEvaluator(
        repository, repo_map=without_symbols_map
    ).evaluate(cases, k=k, split=split)
    wins = ties = losses = 0
    for with_detail, without_detail in zip(
        structured.details, without_symbols.details
    ):
        if with_detail.id != without_detail.id:
            raise ValueError("comparison cases are not aligned")
        if with_detail.reciprocal_rank > without_detail.reciprocal_rank:
            wins += 1
        elif with_detail.reciprocal_rank < without_detail.reciprocal_rank:
            losses += 1
        else:
            ties += 1
    return StructureComparison(
        dataset_sha256=structured.dataset_sha256,
        repository_sha256=repository_digest.hexdigest(),
        cases=structured.cases,
        k=k,
        structured=structured,
        without_symbols=without_symbols,
        recall_delta=round(
            structured.recall_at_k - without_symbols.recall_at_k, 6
        ),
        mrr_delta=round(structured.mrr - without_symbols.mrr, 6),
        structured_wins=wins,
        ties=ties,
        without_symbols_wins=losses,
    )
