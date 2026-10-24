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
    assert result.baseline_mrr == 1.0
    assert result.mrr_delta == 0.0
    assert result.pairwise.ties == 1
    assert result.pairwise.progressive_wins == 0
    assert result.pairwise.baseline_wins == 0
    assert result.bad_cases == []


def test_evaluator_reports_bad_case_reasons(tmp_path: Path):
    (tmp_path / "target.py").write_text("def target():\n    return True\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def other():\n    return True\n", encoding="utf-8")
    cases = [
        RetrievalCase(
            id="missing",
            query="unfindable requirement",
            relevant_paths=["target.py"],
        )
    ]

    result = RetrievalEvaluator(Repository(tmp_path)).evaluate(cases, k=1)

    assert result.bad_cases[0].id == "missing"
    assert "missed_relevant_files" in result.bad_cases[0].reasons
    assert result.bad_cases[0].missing_paths == ["target.py"]


def test_load_cases_rejects_invalid_dataset_shape(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text('{"query": "wrong shape"}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array or an object with cases"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids_queries_and_unsafe_paths(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(
        """[
          {"id": "one", "query": "find quota logic", "relevant_paths": ["quota.py"]},
          {"id": "one", "query": "find other logic", "relevant_paths": ["../other.py"]}
        ]""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ids must be unique"):
        load_cases(path)


def test_load_cases_accepts_versioned_dataset_and_checks_files(tmp_path: Path):
    (tmp_path / "quota.py").write_text("def reserve(): pass\n", encoding="utf-8")
    path = tmp_path / "cases.json"
    path.write_text(
        """{
          "schema_version": 1,
          "dataset_id": "fixture-v1",
          "cases": [
            {"id": "quota", "query": "find reservation behavior", "relevant_paths": ["quota.py"]}
          ]
        }""",
        encoding="utf-8",
    )

    cases = load_cases(path, repository_root=tmp_path)

    assert [case.id for case in cases] == ["quota"]
