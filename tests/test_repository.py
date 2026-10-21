from pathlib import Path

import pytest

from ai_harness.repository import (
    ContextIndex,
    QueryRewriter,
    RepoMapBuilder,
    Repository,
    RepositoryBoundaryError,
    RetrievalCache,
)


def test_repo_map_extracts_python_symbols_and_imports(tmp_path: Path):
    (tmp_path / "service.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "class Gateway:\n"
        "    pass\n\n"
        "async def run_task():\n"
        "    return Path('.')\n",
        encoding="utf-8",
    )
    repo_map = RepoMapBuilder().build(Repository(tmp_path))

    assert len(repo_map) == 1
    assert [symbol.name for symbol in repo_map[0].symbols] == ["Gateway", "run_task"]
    assert repo_map[0].imports == ["json", "pathlib"]


def test_context_search_ranks_symbol_and_text_matches(tmp_path: Path):
    (tmp_path / "quota.py").write_text(
        "class QuotaManager:\n    def reserve_tokens(self):\n        return 'quota reservation'\n",
        encoding="utf-8",
    )
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("fix quota reservation")

    assert hits
    assert hits[0].path == "quota.py"
    assert any("reservation" in hit.text for hit in hits)
    assert hits[0].source.startswith("hybrid_rrf")


def test_context_search_adds_chinese_bigrams(tmp_path: Path):
    (tmp_path / "payment.py").write_text("# 防止重复扣款\n", encoding="utf-8")
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("修复订单重复扣款问题")

    assert hits and hits[0].path == "payment.py"


def test_repository_rejects_path_escape(tmp_path: Path):
    repository = Repository(tmp_path)
    with pytest.raises(RepositoryBoundaryError):
        repository.resolve("../outside.txt", must_exist=False)


def test_repository_honors_file_scope(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("x = 1", encoding="utf-8")

    files = list(Repository(tmp_path).iter_files(["src"]))

    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["src/app.py"]


def test_query_rewriter_expands_domain_terms():
    rewritten = QueryRewriter().rewrite("修复 token 限流 timeout")

    assert any("rpm" in item and "deadline" in item for item in rewritten)


def test_hybrid_retrieval_fuses_structure_lexical_and_git_context(tmp_path: Path):
    (tmp_path / "quota.py").write_text(
        "class QuotaManager:\n    def reserve_tokens(self):\n        return 'reserved'\n",
        encoding="utf-8",
    )
    (tmp_path / "worker.py").write_text("import quota\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("value = 1\n" * 200, encoding="utf-8")
    repository = Repository(tmp_path)
    index = ContextIndex(
        repository,
        RepoMapBuilder().build(repository),
        changed_paths=["quota.py"],
    )

    hits = index.search("reserve quota tokens", limit=2)

    assert hits[0].path == "quota.py"
    assert "lexical" in hits[0].source
    assert index.last_trace.stages[-1].selected_paths[0] == "quota.py"
    assert index.last_trace.context_reduction_ratio > 0


def test_retrieval_cache_hits_and_repository_change_invalidates_it(tmp_path: Path):
    target = tmp_path / "payments.py"
    target.write_text("def reserve_payment():\n    return True\n", encoding="utf-8")
    repository = Repository(tmp_path)
    cache = RetrievalCache()

    first = ContextIndex(repository, RepoMapBuilder().build(repository), cache=cache)
    first.search("reserve payment")
    second = ContextIndex(repository, RepoMapBuilder().build(repository), cache=cache)
    second.search("reserve payment")
    assert second.last_trace.cache_hit is True

    target.write_text("def reserve_payment():\n    return False\n", encoding="utf-8")
    third = ContextIndex(repository, RepoMapBuilder().build(repository), cache=cache)
    third.search("reserve payment")
    assert third.last_trace.cache_hit is False


def test_context_budget_caps_selected_text(tmp_path: Path):
    (tmp_path / "large.py").write_text(
        "\n".join(f"quota token reservation line {index}" for index in range(100)),
        encoding="utf-8",
    )
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("quota token reservation", max_context_chars=80)

    assert sum(len(hit.text) for hit in hits) <= 80
    assert index.last_trace.stop_reason == "budget_exhausted"
