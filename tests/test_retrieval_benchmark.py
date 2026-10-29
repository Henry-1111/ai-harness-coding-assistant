from pathlib import Path

from ai_harness.evaluation import (
    RetrievalEvaluator,
    compare_structure_index,
    load_cases,
    retrieval_cases_sha256,
)
from ai_harness.repository import Repository


def test_labelled_retrieval_benchmark_stays_above_regression_floor():
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(
        root / "benchmarks" / "retrieval_cases.json", repository_root=root
    )
    result = RetrievalEvaluator(Repository(root)).evaluate(
        cases,
        k=5,
        split="test",
    )

    assert len(cases) >= 30
    assert len([case for case in cases if case.split == "test"]) >= 30
    assert len({case.task_type for case in cases}) >= 8
    assert result.dataset_sha256 == retrieval_cases_sha256(cases)
    assert result.dataset_sha256 == (
        "18b616dc8877e8dff54beab6a25c7db6169abb49b71e879a5fa814c2424583c1"
    )
    assert result.recall_at_k >= 0.8
    assert result.mrr >= 0.75
    assert result.ndcg_at_k >= 0.75
    assert result.hit_rate >= 0.9
    assert result.mean_context_reduction_ratio >= 0.9


def test_multilang_retrieval_pilot_compares_structure_with_text_only():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "benchmarks" / "fixtures" / "multilang_retrieval"
    cases = load_cases(
        root / "benchmarks" / "multilang_retrieval_cases.json",
        repository_root=fixture,
    )
    comparison = compare_structure_index(Repository(fixture), cases, k=1)

    assert len(cases) == 8
    assert {case.language for case in cases} == {
        "javascript", "typescript", "go", "java"
    }
    assert comparison.dataset_sha256 == retrieval_cases_sha256(cases)
    assert comparison.dataset_sha256 == (
        "267274715224d469541f84b586b588d7bf0bd015ac16bbb26761478aa8abee45"
    )
    assert comparison.repository_sha256 == (
        "aa8e3b3eaeac43565c4c42d1c981b6afe4212e557143a32d34271588618f3fdc"
    )
    assert comparison.cases == 8
    assert comparison.structured_wins + comparison.ties + comparison.without_symbols_wins == 8
    assert set(comparison.structured.by_language) == {
        "javascript", "typescript", "go", "java"
    }
    assert comparison.structured.recall_at_k >= 0.75
