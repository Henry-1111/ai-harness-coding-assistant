import subprocess
from pathlib import Path

from ai_harness.git_tools import GitRepository


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


def test_git_context_reports_head_history_and_changes(tmp_path: Path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    git(tmp_path, "add", "app.py")
    git(tmp_path, "commit", "-m", "initial")
    repository = GitRepository(tmp_path)

    assert len(repository.head()) == 40
    assert repository.branch() == "main"
    assert repository.recent_history(1)[0].endswith("\tinitial")

    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    assert repository.changes()[0].path == "app.py"
    assert "-value = 1" in repository.diff("app.py")
    assert "+value = 2" in repository.diff("app.py")
