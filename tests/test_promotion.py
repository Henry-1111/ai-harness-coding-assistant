from __future__ import annotations

import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from ai_harness.promotion import (
    GitBranchPublisher,
    GitHubPullRequestClient,
    GitHubPullRequestError,
    GitHubPullRequestPublisher,
    PromotionConflictError,
)


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

    with pytest.raises(PromotionConflictError, match="different content"):
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


def test_github_publisher_creates_then_reuses_pull_request(tmp_path: Path) -> None:
    remote, workspace = remote_and_workspace(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and len(requests) == 1:
            return httpx.Response(200, json=[])
        if request.method == "POST":
            payload = json.loads(request.content)
            assert payload == {
                "title": "AI Harness task task-123",
                "head": "ai-harness/task-123",
                "base": "main",
                "body": (
                    "Automated branch produced by AI Harness after verification "
                    "and reviewer approval."
                ),
                "draft": True,
            }
            return httpx.Response(
                201,
                json={"html_url": "https://github.com/acme/project/pull/7", "number": 7},
            )
        return httpx.Response(
            200,
            json=[{"html_url": "https://github.com/acme/project/pull/7", "number": 7}],
        )

    github = GitHubPullRequestClient(
        "acme/project",
        "secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    publisher = GitHubPullRequestPublisher(
        GitBranchPublisher(str(remote), tmp_path / "staging"), github
    )

    created = publisher.publish("task-123", workspace, ["app.py"])
    replay = publisher.publish("task-123", workspace, ["app.py"])

    assert created.pull_request_url == "https://github.com/acme/project/pull/7"
    assert created.pull_request_number == 7
    assert created.pull_request_created is True
    assert replay.pull_request_created is False
    assert replay.commit == created.commit
    assert len(requests) == 3
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    assert requests[0].headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert requests[0].url.params["head"] == "acme:ai-harness/task-123"
    assert "secret-token" not in repr(created)


def test_github_pull_request_creation_race_reuses_winner() -> None:
    responses = iter(
        [
            httpx.Response(200, json=[]),
            httpx.Response(422, json={"message": "already exists"}),
            httpx.Response(
                200,
                json=[
                    {"html_url": "https://github.com/acme/project/pull/9", "number": 9}
                ],
            ),
        ]
    )
    github = GitHubPullRequestClient(
        "acme/project",
        "secret-token",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))),
    )

    receipt = github.ensure("task-123", "ai-harness/task-123", "main")

    assert receipt.number == 9
    assert receipt.created is False


def test_github_pull_request_error_redacts_token() -> None:
    github = GitHubPullRequestClient(
        "acme/project",
        "super-secret",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    403, text="credential super-secret cannot create pull requests"
                )
            )
        ),
    )

    with pytest.raises(GitHubPullRequestError) as captured:
        github.ensure("task-123", "ai-harness/task-123", "main")

    assert "super-secret" not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)
