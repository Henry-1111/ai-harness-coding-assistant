from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from .repository import Repository

BRANCH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")


class GitPublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromotionReceipt:
    mode: str
    branch: str
    commit: str
    remote: str
    created: bool


class PromotionPublisher(Protocol):
    def publish(
        self,
        task_id: str,
        workspace: Path,
        changed_files: Sequence[str],
    ) -> PromotionReceipt: ...


class GitBranchPublisher:
    """Publish reviewed files to a unique remote branch without touching main."""

    def __init__(
        self,
        remote_url: str,
        staging_root: Path,
        *,
        base_ref: str = "main",
        branch_prefix: str = "ai-harness",
        author_name: str = "AI Harness",
        author_email: str = "ai-harness@localhost",
        timeout_seconds: float = 120,
    ) -> None:
        if not remote_url.strip():
            raise ValueError("Git promotion remote URL is required")
        if not self._valid_ref(base_ref):
            raise ValueError("invalid Git promotion base ref")
        prefix = branch_prefix.strip("/")
        if not self._valid_ref(prefix):
            raise ValueError("invalid Git promotion branch prefix")
        if timeout_seconds <= 0:
            raise ValueError("Git promotion timeout must be positive")
        self.remote_url = remote_url.strip()
        self.remote_label = self._safe_remote_label(self.remote_url)
        self.staging_root = staging_root.resolve()
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.base_ref = base_ref
        self.branch_prefix = prefix
        self.author_name = author_name
        self.author_email = author_email
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _valid_ref(value: str) -> bool:
        return bool(BRANCH_COMPONENT_PATTERN.fullmatch(value)) and not any(
            (
                value.startswith("."),
                value.endswith("."),
                value.endswith(".lock"),
                ".." in value,
                "//" in value,
                "@{" in value,
            )
        )

    @staticmethod
    def _safe_remote_label(remote_url: str) -> str:
        parsed = urlsplit(remote_url)
        if not parsed.scheme or not parsed.netloc:
            return remote_url
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))

    def _run(self, *arguments: str, cwd: Path | None = None) -> str:
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            message = message.replace(self.remote_url, self.remote_label)
            raise GitPublishError(message or "Git promotion command failed")
        return completed.stdout.strip()

    def _branch_name(self, task_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", task_id):
            raise ValueError("invalid task id for Git promotion")
        return f"{self.branch_prefix}/{task_id}"

    def _remote_branch_commit(self, branch: str) -> str | None:
        completed = subprocess.run(
            ["git", "ls-remote", "--heads", self.remote_url, f"refs/heads/{branch}"],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip().replace(self.remote_url, self.remote_label)
            raise GitPublishError(message or "cannot inspect Git remote")
        output = completed.stdout.strip()
        return output.split()[0] if output else None

    def _existing_receipt(
        self,
        branch: str,
        workspace: Repository,
        changed_files: Sequence[str],
        checkout_path: Path,
    ) -> PromotionReceipt | None:
        """Return an idempotent receipt when the immutable branch already exists."""
        if self._remote_branch_commit(branch) is None:
            return None
        self._run(
            "clone",
            "--quiet",
            "--branch",
            branch,
            "--single-branch",
            self.remote_url,
            str(checkout_path),
        )
        checkout = Repository(checkout_path)
        if not self._matches_workspace(checkout, workspace, changed_files):
            raise GitPublishError(
                f"remote promotion branch already exists with different content: {branch}"
            )
        return PromotionReceipt(
            mode="git_branch",
            branch=branch,
            commit=self._run("rev-parse", "HEAD", cwd=checkout_path),
            remote=self.remote_label,
            created=False,
        )

    @staticmethod
    def _matches_workspace(
        checkout: Repository,
        workspace: Repository,
        changed_files: Sequence[str],
    ) -> bool:
        for relative_path in changed_files:
            expected = workspace.resolve(relative_path).read_bytes()
            try:
                actual = checkout.resolve(relative_path).read_bytes()
            except FileNotFoundError:
                return False
            if actual != expected:
                return False
        return True

    def publish(
        self,
        task_id: str,
        workspace: Path,
        changed_files: Sequence[str],
    ) -> PromotionReceipt:
        if not changed_files:
            raise GitPublishError("there are no reviewed files to publish")
        branch = self._branch_name(task_id)
        workspace_repository = Repository(workspace)
        for relative_path in changed_files:
            workspace_repository.resolve(relative_path)
        with tempfile.TemporaryDirectory(
            prefix=".git-promotion-", dir=self.staging_root
        ) as directory:
            staging_directory = Path(directory)
            existing = self._existing_receipt(
                branch,
                workspace_repository,
                changed_files,
                staging_directory / "existing",
            )
            if existing is not None:
                return existing
            checkout_path = staging_directory / "checkout"
            self._run(
                "clone",
                "--quiet",
                "--branch",
                self.base_ref,
                "--single-branch",
                self.remote_url,
                str(checkout_path),
            )
            checkout = Repository(checkout_path)
            self._run("checkout", "-q", "-b", branch, cwd=checkout_path)
            for relative_path in changed_files:
                source = workspace_repository.resolve(relative_path)
                destination = checkout.resolve(relative_path, must_exist=False)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            self._run("add", "--", *changed_files, cwd=checkout_path)
            self._run(
                "-c",
                f"user.name={self.author_name}",
                "-c",
                f"user.email={self.author_email}",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                f"AI Harness task {task_id}",
                cwd=checkout_path,
            )
            commit = self._run("rev-parse", "HEAD", cwd=checkout_path)
            try:
                self._run(
                    "push",
                    "--porcelain",
                    "origin",
                    f"HEAD:refs/heads/{branch}",
                    cwd=checkout_path,
                )
            except GitPublishError as push_error:
                # Another API instance may win between ls-remote and push. Re-read
                # the immutable branch and settle an equivalent result as a replay.
                raced = self._existing_receipt(
                    branch,
                    workspace_repository,
                    changed_files,
                    staging_directory / "raced",
                )
                if raced is not None:
                    return raced
                raise push_error
        return PromotionReceipt(
            mode="git_branch",
            branch=branch,
            commit=commit,
            remote=self.remote_label,
            created=True,
        )
