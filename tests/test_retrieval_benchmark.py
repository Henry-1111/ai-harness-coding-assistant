from pathlib import Path

from ai_harness.evaluation import RetrievalEvaluator, load_cases, retrieval_cases_sha256
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
