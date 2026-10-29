from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field

from .evaluation import RetrievalCase, StructureComparison, compare_structure_index
from .repository import Repository


class RealRepositorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]+$")
    github: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    language: str
    repository_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    cases: List[RetrievalCase] = Field(min_length=1)


class RealRepositoryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    dataset_id: str
    scope: str
    repositories: List[RealRepositorySpec] = Field(min_length=1)


class RealRepositoryResult(BaseModel):
    id: str
    github: str
    commit: str
    language: str
    comparison: StructureComparison


class RealRepositoryReport(BaseModel):
    dataset_id: str
    manifest_sha256: str
    k: int
    cases: int
    structured_recall_at_k: float
    without_symbols_recall_at_k: float
    structured_mrr: float
    without_symbols_mrr: float
    structured_wins: int
    ties: int
    without_symbols_wins: int
    repositories: List[RealRepositoryResult]


def load_real_repo_manifest(path: Path) -> RealRepositoryManifest:
    manifest = RealRepositoryManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.schema_version != 1:
        raise ValueError("unsupported real-repository benchmark schema")
    ids = [spec.id for spec in manifest.repositories]
    if len(ids) != len(set(ids)):
        raise ValueError("repository ids must be unique")
    case_ids = [case.id for spec in manifest.repositories for case in spec.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case ids must be unique across repositories")
    for spec in manifest.repositories:
        for case in spec.cases:
            if case.language != spec.language:
                raise ValueError(f"case language does not match repository: {case.id}")
    return manifest


def evaluate_real_repositories(
    manifest_path: Path, roots: Dict[str, Path], *, k: int = 3
) -> RealRepositoryReport:
    manifest = load_real_repo_manifest(manifest_path)
    expected_ids = {spec.id for spec in manifest.repositories}
    if set(roots) != expected_ids:
        raise ValueError("repository roots must match manifest ids exactly")
    results: List[RealRepositoryResult] = []
    for spec in manifest.repositories:
        root = roots[spec.id].resolve()
        repository = Repository(root)
        if (root / ".git").exists():
            actual_commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual_commit != spec.commit:
                raise ValueError(f"wrong commit for {spec.id}: {actual_commit}")
        for case in spec.cases:
            if len(case.relevant_paths) != len(set(case.relevant_paths)):
                raise ValueError(f"duplicate relevant path: {case.id}")
            for relative in case.relevant_paths:
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError(f"unsafe relevant path: {case.id}")
                if relative.lower() in case.query.lower():
                    raise ValueError(f"query leaks target path: {case.id}")
                if not repository.resolve(relative).is_file():
                    raise ValueError(f"missing relevant file: {case.id}: {relative}")
        comparison = compare_structure_index(repository, spec.cases, k=k)
        if comparison.repository_sha256 != spec.repository_sha256:
            raise ValueError(f"repository content changed for {spec.id}")
        results.append(
            RealRepositoryResult(
                id=spec.id,
                github=spec.github,
                commit=spec.commit,
                language=spec.language,
                comparison=comparison,
            )
        )
    total = sum(result.comparison.cases for result in results)

    def weighted(metric: str, variant: str) -> float:
        return round(
            sum(
                getattr(getattr(result.comparison, variant), metric)
                * result.comparison.cases
                for result in results
            ) / total,
            6,
        )

    canonical = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return RealRepositoryReport(
        dataset_id=manifest.dataset_id,
        manifest_sha256=sha256(canonical).hexdigest(),
        k=k,
        cases=total,
        structured_recall_at_k=weighted("recall_at_k", "structured"),
        without_symbols_recall_at_k=weighted("recall_at_k", "without_symbols"),
        structured_mrr=weighted("mrr", "structured"),
        without_symbols_mrr=weighted("mrr", "without_symbols"),
        structured_wins=sum(result.comparison.structured_wins for result in results),
        ties=sum(result.comparison.ties for result in results),
        without_symbols_wins=sum(
            result.comparison.without_symbols_wins for result in results
        ),
        repositories=results,
    )
