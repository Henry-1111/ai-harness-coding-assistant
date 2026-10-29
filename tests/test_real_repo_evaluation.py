import json
from pathlib import Path

import pytest

from ai_harness.evaluation import RetrievalCase, compare_structure_index
from ai_harness.real_repo_evaluation import (
    evaluate_real_repositories,
    load_real_repo_manifest,
)
from ai_harness.repository import Repository


def test_real_repo_manifest_keeps_four_languages_and_twelve_cases():
    root = Path(__file__).resolve().parents[1]
    manifest = load_real_repo_manifest(
        root / "benchmarks" / "real_repo_retrieval_cases.json"
    )

    assert len(manifest.repositories) == 4
    assert {spec.language for spec in manifest.repositories} == {
        "javascript", "typescript", "go", "java"
    }
    assert sum(len(spec.cases) for spec in manifest.repositories) == 12
    assert all(len(spec.commit) == 40 for spec in manifest.repositories)
    assert all(len(spec.repository_sha256) == 64 for spec in manifest.repositories)

    holdout = load_real_repo_manifest(
        root / "benchmarks" / "real_repo_retrieval_holdout.json"
    )
    assert sum(len(spec.cases) for spec in holdout.repositories) == 8
    assert {spec.id: spec.commit for spec in holdout.repositories} == {
        spec.id: spec.commit for spec in manifest.repositories
    }
    assert {case.id for spec in manifest.repositories for case in spec.cases}.isdisjoint(
        {case.id for spec in holdout.repositories for case in spec.cases}
    )

    issues = load_real_repo_manifest(
        root / "benchmarks" / "real_issue_retrieval_cases.json"
    )
    assert len(issues.repositories) == 4
    assert sum(len(spec.cases) for spec in issues.repositories) == 4
    assert {spec.language for spec in issues.repositories} == {
        "javascript", "typescript", "go", "java"
    }
    assert all(spec.source_url and spec.fix_url for spec in issues.repositories)
    assert {case.id for spec in issues.repositories for case in spec.cases}.isdisjoint(
        {case.id for spec in manifest.repositories for case in spec.cases}
        | {case.id for spec in holdout.repositories for case in spec.cases}
    )


def test_real_repo_runner_checks_content_and_scores_without_network(tmp_path: Path):
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    (repository_root / "implementation.js").write_text(
        "export function resolveOwner(id) { return id; }\n", encoding="utf-8"
    )
    case = RetrievalCase(
        id="resolve-owner",
        query="find the function that resolves the owner by identifier",
        relevant_paths=["implementation.js"],
        language="javascript",
    )
    digest = compare_structure_index(Repository(repository_root), [case], k=1)
    manifest = {
        "schema_version": 1,
        "dataset_id": "unit-test",
        "scope": "unit test",
        "repositories": [
            {
                "id": "example",
                "github": "https://github.com/example/example",
                "commit": "a" * 40,
                "language": "javascript",
                "repository_sha256": digest.repository_sha256,
                "cases": [case.model_dump(mode="json")],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_real_repositories(
        manifest_path, {"example": repository_root}, k=1
    )
    assert report.cases == 1
    assert report.structured_recall_at_k == 1.0
    assert report.repositories[0].comparison.repository_sha256 == digest.repository_sha256
    assert report.repositories[0].comparison.structured.details[0].stage_relevant_ranks

    (repository_root / "implementation.js").write_text(
        "export function changed() { return true; }\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="repository content changed"):
        evaluate_real_repositories(manifest_path, {"example": repository_root}, k=1)
