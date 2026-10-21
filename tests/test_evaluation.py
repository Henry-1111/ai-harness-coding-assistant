from pathlib import Path

import pytest

from ai_harness.evaluation import RetrievalCase, RetrievalEvaluator, load_cases
from ai_harness.repository import Repository


def test_retrieval_evaluator_reports_recall_precision_mrr_and_reduction(tmp_path: Path):
    (tmp_path / "quota.py").write_text(
        "def reserve_token_quota():\n    return 'reservation'\n", encoding="utf-8"
    )
    (tmp_path / "unrelated.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    cases = [
        RetrievalCase(
            id="quota",
            query="reserve token quota reservation",
            relevant_paths=["quota.py"],
            language="python",
            task_type="quota",
        )
    ]

    result = RetrievalEvaluator(Repository(tmp_path)).evaluate(cases, k=2)

    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 0.5
    assert result.mrr == 1.0
    assert result.ndcg_at_k == 1.0
    assert result.hit_rate == 1.0
    assert result.mean_context_reduction_ratio > 0
    assert result.by_language["python"].recall_at_k == 1.0
    assert result.by_task_type["quota"].hit_rate == 1.0
    assert result.baseline_recall_at_k == 1.0


def test_load_cases_rejects_non_array_dataset(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text('{"query": "wrong shape"}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        load_cases(path)
