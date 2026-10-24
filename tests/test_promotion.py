from __future__ import annotations

import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ai_harness.promotion import GitBranchPublisher, GitPublishError


def git(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def remote_and_workspace(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    git("init", "-q", "-b", "main", cwd=source)
    git("config", "user.name", "Test", cwd=source)
    git("config", "user.email", "test@example.com", cwd=source)
    (source / "app.py").write_text("old\n")
    git("add", "app.py", cwd=source)
    git("commit", "-q", "-m", "initial", cwd=source)
    remote = tmp_path / "remote.git"
    git("clone", "-q", "--bare", str(source), str(remote))
    workspace = tmp_path / "workspace"
    shutil.copytree(source, workspace, ignore=shutil.ignore_patterns(".git"))
    (workspace / "app.py").write_text("reviewed\n")
    return remote, workspace


def test_git_branch_publisher_creates_idempotent_remote_branch(tmp_path: Path) -> None:
    remote, workspace = remote_and_workspace(tmp_path)
    publisher = GitBranchPublisher(str(remote), tmp_path / "staging")

    created = publisher.publish("task-123", workspace, ["app.py"])
    replay = publisher.publish("task-123", workspace, ["app.py"])

    assert created.mode == "git_branch"
    assert created.branch == "ai-harness/task-123"
    assert created.created is True
    assert replay.created is False
    assert replay.commit == created.commit
    checkout = tmp_path / "checkout"
    git("clone", "-q", "--branch", created.branch, str(remote), str(checkout))
    assert (checkout / "app.py").read_text() == "reviewed\n"


def test_git_branch_publisher_never_overwrites_conflicting_remote_branch(
    tmp_path: Path,
) -> None:
    remote, workspace = remote_and_workspace(tmp_path)
    publisher = GitBranchPublisher(str(remote), tmp_path / "staging")
    publisher.publish("task-123", workspace, ["app.py"])
    (workspace / "app.py").write_text("different result\n")

    with pytest.raises(GitPublishError, match="different content"):
        publisher.publish("task-123", workspace, ["app.py"])

    checkout = tmp_path / "checkout"
    git("clone", "-q", "--branch", "ai-harness/task-123", str(remote), str(checkout))
    assert (checkout / "app.py").read_text() == "reviewed\n"


def test_concurrent_equivalent_publish_settles_as_idempotent_replay(
    tmp_path: Path,
) -> None:
    remote, workspace = remote_and_workspace(tmp_path)
    barrier = threading.Barrier(2)

    class RacingPublisher(GitBranchPublisher):
        def _remote_branch_commit(self, branch: str) -> str | None:
            commit = super()._remote_branch_commit(branch)
            if commit is None:
                barrier.wait(timeout=5)
            return commit

    publishers = [
        RacingPublisher(
            str(remote),
            tmp_path / f"staging-{index}",
            author_name=f"Publisher {index}",
        )
        for index in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(
            executor.map(
                lambda publisher: publisher.publish(
                    "task-concurrent", workspace, ["app.py"]
                ),
                publishers,
            )
        )

    assert sorted(receipt.created for receipt in receipts) == [False, True]
    assert receipts[0].commit == receipts[1].commit
    checkout = tmp_path / "concurrent-checkout"
    git(
        "clone",
        "-q",
        "--branch",
        "ai-harness/task-concurrent",
        str(remote),
        str(checkout),
    )
    assert (checkout / "app.py").read_text() == "reviewed\n"


def test_git_branch_publisher_rejects_unsafe_task_id(tmp_path: Path) -> None:
    remote, workspace = remote_and_workspace(tmp_path)
    publisher = GitBranchPublisher(str(remote), tmp_path / "staging")

    with pytest.raises(ValueError, match="invalid task id"):
        publisher.publish("../main", workspace, ["app.py"])


def test_git_branch_publisher_redacts_url_credentials_from_receipt(tmp_path: Path) -> None:
    publisher = GitBranchPublisher(
        "https://secret-token@example.com/org/repo.git", tmp_path / "staging"
    )

    assert publisher.remote_label == "https://example.com/org/repo.git"


@pytest.mark.parametrize("prefix", ["../escape", ".hidden", "branch..name", "a//b"])
def test_git_branch_publisher_rejects_unsafe_branch_prefix(
    tmp_path: Path, prefix: str
) -> None:
    with pytest.raises(ValueError, match="invalid Git promotion branch prefix"):
        GitBranchPublisher("remote.git", tmp_path / "staging", branch_prefix=prefix)
