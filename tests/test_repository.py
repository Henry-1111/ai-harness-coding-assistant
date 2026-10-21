from pathlib import Path

import pytest

from ai_harness.repository import ContextIndex, RepoMapBuilder, Repository, RepositoryBoundaryError


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
    assert hits[0].source == "text_window"


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
