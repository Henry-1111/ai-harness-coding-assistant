from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from .repository import Repository

BRANCH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")
GITHUB_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
GITLAB_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class GitPublishError(RuntimeError):
    pass


class PromotionConflictError(GitPublishError):
    """The immutable remote branch exists with non-equivalent content."""

    pass


class GitHubPullRequestError(GitPublishError):
    pass


class GitLabMergeRequestError(GitPublishError):
    pass


@dataclass(frozen=True)
class PromotionReceipt:
    mode: str
    branch: str
    commit: str
    remote: str
    created: bool
    pull_request_url: str | None = None
    pull_request_number: int | None = None
    pull_request_created: bool | None = None


class PromotionPublisher(Protocol):
    def publish(
        self,
        task_id: str,
        workspace: Path,
        changed_files: Sequence[str],
    ) -> PromotionReceipt: ...


@dataclass(frozen=True)
class PullRequestReceipt:
    url: str
    number: int
    created: bool


class GitHubPullRequestClient:
    """Create or reuse one GitHub pull request for an immutable task branch."""

    def __init__(
        self,
        repository: str,
        token: str,
        *,
        api_url: str = "https://api.github.com",
        api_version: str = "2022-11-28",
        timeout_seconds: float = 30,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
        max_retry_after_seconds: float = 5,
        draft: bool = True,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("invalid GitHub repository; expected owner/name")
        owner, repository_name = repository.split("/", 1)
        if owner in {".", ".."} or repository_name in {".", ".."}:
            raise ValueError("invalid GitHub repository; expected owner/name")
        if not token.strip():
            raise ValueError("GitHub pull request token is required")
        parsed = urlsplit(api_url.rstrip("/"))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid GitHub API URL")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", api_version):
            raise ValueError("invalid GitHub API version")
        if timeout_seconds <= 0:
            raise ValueError("GitHub API timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("GitHub API max attempts must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("GitHub API retry backoff must not be negative")
        if max_retry_after_seconds < 0:
            raise ValueError("GitHub API max retry-after must not be negative")
        self.owner = owner
        self.repository_name = repository_name
        self.token = token.strip()
        self.api_url = api_url.rstrip("/")
        self.api_version = api_version
        self.draft = draft
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_retry_after_seconds = max_retry_after_seconds
        self.client = client
        self.sleeper = sleeper

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "ai-harness",
            "X-GitHub-Api-Version": self.api_version,
        }

    def _pulls_url(self) -> str:
        owner = quote(self.owner, safe="")
        repository = quote(self.repository_name, safe="")
        return f"{self.api_url}/repos/{owner}/{repository}/pulls"

    def _request(self, method: str, **kwargs: Any) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.client is not None:
                    response = self.client.request(method, self._pulls_url(), **kwargs)
                else:
                    with httpx.Client(timeout=self.timeout_seconds) as client:
                        response = client.request(method, self._pulls_url(), **kwargs)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    break
                self.sleeper(self._retry_delay(attempt))
                continue
            if (
                response.status_code not in GITHUB_RETRYABLE_STATUS_CODES
                or attempt == self.max_attempts
            ):
                return response
            self.sleeper(self._retry_delay(attempt, response))
        assert last_error is not None
        raise GitHubPullRequestError(
            f"GitHub pull request API request failed: {last_error}"
        ) from last_error

    def _retry_delay(
        self, attempt: int, response: httpx.Response | None = None
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return min(
                        max(float(retry_after), 0), self.max_retry_after_seconds
                    )
                except ValueError:
                    pass
        return min(
            self.retry_backoff_seconds * (2 ** (attempt - 1)),
            self.max_retry_after_seconds,
        )

    def _error(self, response: httpx.Response) -> GitHubPullRequestError:
        message = response.text.strip().replace(self.token, "[REDACTED]")
        if len(message) > 500:
            message = message[:500] + "..."
        return GitHubPullRequestError(
            f"GitHub pull request API returned {response.status_code}: "
            f"{message or response.reason_phrase}"
        )

    @staticmethod
    def _receipt(payload: object, *, created: bool) -> PullRequestReceipt:
        if not isinstance(payload, dict):
            raise GitHubPullRequestError("GitHub pull request API returned invalid JSON")
        url = payload.get("html_url")
        number = payload.get("number")
        if (
            not isinstance(url, str)
            or not url
            or not isinstance(number, int)
            or isinstance(number, bool)
        ):
            raise GitHubPullRequestError("GitHub pull request API response is incomplete")
        return PullRequestReceipt(url=url, number=number, created=created)

    def _find_existing(self, branch: str, base_ref: str) -> PullRequestReceipt | None:
        response = self._request(
            "GET",
            headers=self._headers(),
            params={
                "state": "open",
                "head": f"{self.owner}:{branch}",
                "base": base_ref,
                "per_page": 100,
            },
        )
        if response.status_code != 200:
            raise self._error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GitHubPullRequestError(
                "GitHub pull request API returned invalid JSON"
            ) from exc
        if not isinstance(payload, list):
            raise GitHubPullRequestError("GitHub pull request API returned invalid JSON")
        return self._receipt(payload[0], created=False) if payload else None

    def ensure(self, task_id: str, branch: str, base_ref: str) -> PullRequestReceipt:
        existing = self._find_existing(branch, base_ref)
        if existing is not None:
            return existing
        response = self._request(
            "POST",
            headers=self._headers(),
            json={
                "title": f"AI Harness task {task_id}",
                "head": branch,
                "base": base_ref,
                "body": (
                    "Automated branch produced by AI Harness after verification "
                    "and reviewer approval."
                ),
                "draft": self.draft,
            },
        )
        if response.status_code == 201:
            try:
                return self._receipt(response.json(), created=True)
            except ValueError as exc:
                raise GitHubPullRequestError(
                    "GitHub pull request API returned invalid JSON"
                ) from exc
        if response.status_code == 422:
            # Another instance can create the PR after our initial list request.
            raced = self._find_existing(branch, base_ref)
            if raced is not None:
                return raced
        raise self._error(response)


class GitHubPullRequestPublisher:
    """Publish a branch, then create or reuse its GitHub pull request."""

    def __init__(
        self,
        branch_publisher: GitBranchPublisher,
        pull_requests: GitHubPullRequestClient,
    ) -> None:
        self.branch_publisher = branch_publisher
        self.pull_requests = pull_requests

    def publish(
        self,
        task_id: str,
        workspace: Path,
        changed_files: Sequence[str],
    ) -> PromotionReceipt:
        branch = self.branch_publisher.publish(task_id, workspace, changed_files)
        pull_request = self.pull_requests.ensure(
            task_id, branch.branch, self.branch_publisher.base_ref
        )
        return PromotionReceipt(
            mode=branch.mode,
            branch=branch.branch,
            commit=branch.commit,
            remote=branch.remote,
            created=branch.created,
            pull_request_url=pull_request.url,
            pull_request_number=pull_request.number,
            pull_request_created=pull_request.created,
        )


class GitLabMergeRequestClient:
    """Create or reuse one GitLab merge request for an immutable task branch."""

    def __init__(
        self,
        project: str,
        token: str,
        *,
        api_url: str = "https://gitlab.com/api/v4",
        timeout_seconds: float = 30,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
        max_retry_after_seconds: float = 5,
        draft: bool = True,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        components = project.split("/")
        if not components or any(
            component in {"", ".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", component)
            for component in components
        ):
            raise ValueError("invalid GitLab project; expected numeric ID or group/project")
        if not token.strip():
            raise ValueError("GitLab merge request token is required")
        parsed = urlsplit(api_url.rstrip("/"))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid GitLab API URL")
        if timeout_seconds <= 0:
            raise ValueError("GitLab API timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("GitLab API max attempts must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("GitLab API retry backoff must not be negative")
        if max_retry_after_seconds < 0:
            raise ValueError("GitLab API max retry-after must not be negative")
        self.project = project
        self.token = token.strip()
        self.api_url = api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_retry_after_seconds = max_retry_after_seconds
        self.draft = draft
        self.client = client
        self.sleeper = sleeper

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "PRIVATE-TOKEN": self.token,
            "User-Agent": "ai-harness",
        }

    def _merge_requests_url(self) -> str:
        return f"{self.api_url}/projects/{quote(self.project, safe='')}/merge_requests"

    def _request(self, method: str, **kwargs: Any) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.client is not None:
                    response = self.client.request(
                        method, self._merge_requests_url(), **kwargs
                    )
                else:
                    with httpx.Client(timeout=self.timeout_seconds) as client:
                        response = client.request(
                            method, self._merge_requests_url(), **kwargs
                        )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    break
                self.sleeper(self._retry_delay(attempt))
                continue
            if (
                response.status_code not in GITLAB_RETRYABLE_STATUS_CODES
                or attempt == self.max_attempts
            ):
                return response
            self.sleeper(self._retry_delay(attempt, response))
        assert last_error is not None
        raise GitLabMergeRequestError(
            f"GitLab merge request API request failed: {last_error}"
        ) from last_error

    def _retry_delay(
        self, attempt: int, response: httpx.Response | None = None
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return min(
                        max(float(retry_after), 0), self.max_retry_after_seconds
                    )
                except ValueError:
                    pass
        return min(
            self.retry_backoff_seconds * (2 ** (attempt - 1)),
            self.max_retry_after_seconds,
        )

    def _error(self, response: httpx.Response) -> GitLabMergeRequestError:
        message = response.text.strip().replace(self.token, "[REDACTED]")
        if len(message) > 500:
            message = message[:500] + "..."
        return GitLabMergeRequestError(
            f"GitLab merge request API returned {response.status_code}: "
            f"{message or response.reason_phrase}"
        )

    @staticmethod
    def _receipt(payload: object, *, created: bool) -> PullRequestReceipt:
        if not isinstance(payload, dict):
            raise GitLabMergeRequestError("GitLab merge request API returned invalid JSON")
        url = payload.get("web_url")
        number = payload.get("iid")
        if (
            not isinstance(url, str)
            or not url
            or not isinstance(number, int)
            or isinstance(number, bool)
        ):
            raise GitLabMergeRequestError(
                "GitLab merge request API response is incomplete"
            )
        return PullRequestReceipt(url=url, number=number, created=created)

    def _find_existing(self, branch: str, base_ref: str) -> PullRequestReceipt | None:
        response = self._request(
            "GET",
            headers=self._headers(),
            params={
                "state": "opened",
                "source_branch": branch,
                "target_branch": base_ref,
                "per_page": 100,
            },
        )
        if response.status_code != 200:
            raise self._error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GitLabMergeRequestError(
                "GitLab merge request API returned invalid JSON"
            ) from exc
        if not isinstance(payload, list):
            raise GitLabMergeRequestError("GitLab merge request API returned invalid JSON")
        return self._receipt(payload[0], created=False) if payload else None

    def ensure(self, task_id: str, branch: str, base_ref: str) -> PullRequestReceipt:
        existing = self._find_existing(branch, base_ref)
        if existing is not None:
            return existing
        title = f"AI Harness task {task_id}"
        response = self._request(
            "POST",
            headers=self._headers(),
            json={
                "source_branch": branch,
                "target_branch": base_ref,
                "title": f"Draft: {title}" if self.draft else title,
                "description": (
                    "Automated branch produced by AI Harness after verification "
                    "and reviewer approval."
                ),
                "remove_source_branch": False,
            },
        )
        if response.status_code == 201:
            try:
                return self._receipt(response.json(), created=True)
            except ValueError as exc:
                raise GitLabMergeRequestError(
                    "GitLab merge request API returned invalid JSON"
                ) from exc
        if response.status_code in {400, 409}:
            # A concurrent instance or a retried request may have created the MR.
            raced = self._find_existing(branch, base_ref)
            if raced is not None:
                return raced
        raise self._error(response)


class GitLabMergeRequestPublisher:
    """Publish a branch, then create or reuse its GitLab merge request."""

    def __init__(
        self,
        branch_publisher: GitBranchPublisher,
        merge_requests: GitLabMergeRequestClient,
    ) -> None:
        self.branch_publisher = branch_publisher
        self.merge_requests = merge_requests

    def publish(
        self,
        task_id: str,
        workspace: Path,
        changed_files: Sequence[str],
    ) -> PromotionReceipt:
        branch = self.branch_publisher.publish(task_id, workspace, changed_files)
        merge_request = self.merge_requests.ensure(
            task_id, branch.branch, self.branch_publisher.base_ref
        )
        return PromotionReceipt(
            mode=branch.mode,
            branch=branch.branch,
            commit=branch.commit,
            remote=branch.remote,
            created=branch.created,
            pull_request_url=merge_request.url,
            pull_request_number=merge_request.number,
            pull_request_created=merge_request.created,
        )


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
            raise PromotionConflictError(
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
