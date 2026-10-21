from pathlib import Path

from ai_harness.evaluation import RetrievalEvaluator, load_cases
from ai_harness.repository import Repository


def test_labelled_retrieval_benchmark_stays_above_regression_floor():
    root = Path(__file__).resolve().parents[1]
    result = RetrievalEvaluator(Repository(root)).evaluate(
        load_cases(root / "benchmarks" / "retrieval_cases.json"),
        k=5,
        split="test",
    )

    assert result.recall_at_k >= 0.875
    assert result.mrr >= 0.8
    assert result.ndcg_at_k >= 0.8
    assert result.hit_rate >= 0.875
    assert result.mean_context_reduction_ratio >= 0.9
