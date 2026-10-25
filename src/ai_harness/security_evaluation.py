from __future__ import annotations

import io
import json
import tarfile
import tempfile
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import FileEdit
from .object_workspaces import S3WorkspaceSnapshotStore
from .repository import Repository
from .sandbox import DockerSandbox, validate_command
from .tools import ToolContext, apply_edit
from .workspaces import validate_snapshot_id

AttackCategory = Literal[
    "repository_path",
    "scoped_write",
    "command_allowlist",
    "snapshot_id",
    "archive_path",
    "archive_symlink",
    "docker_policy",
]


class SecurityAttackCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    category: AttackCategory
    payload: str


class SecurityAttackResult(BaseModel):
    id: str
    category: AttackCategory
    blocked: bool
    side_effect_detected: bool
    error: str = ""


class SecurityCategorySummary(BaseModel):
    attacks: int
    blocked: int
    block_rate: float
    side_effects: int


class SecurityBenchmarkReport(BaseModel):
    dataset_sha256: str
    attacks: int
    blocked: int
    block_rate: float
    unauthorized_writes: int
    sandbox_escapes: int
    by_category: Dict[str, SecurityCategorySummary]
    failures: List[SecurityAttackResult]
    details: List[SecurityAttackResult]


def load_security_cases(path: Path) -> List[SecurityAttackCase]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("cases"), list):
        value = value["cases"]
    elif not isinstance(value, list):
        raise ValueError("security dataset must be a JSON array or an object with cases")
    cases = [SecurityAttackCase.model_validate(item) for item in value]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("security attack case ids must be unique")
    return cases


def security_cases_sha256(cases: List[SecurityAttackCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class SecurityEvaluator:
    """Run bounded local attacks and assert that no host-side write escapes occur."""

    def evaluate(self, cases: List[SecurityAttackCase]) -> SecurityBenchmarkReport:
        if not cases:
            raise ValueError("at least one security attack case is required")
        with tempfile.TemporaryDirectory(prefix="ai-harness-security-") as directory:
            root = Path(directory)
            repository_root = root / "repository"
            repository_root.mkdir()
            sentinel = root / "sentinel.txt"
            sentinel.write_text("safe", encoding="utf-8")
            details = [
                self._run_case(case, repository_root=repository_root, sentinel=sentinel)
                for case in cases
            ]
        groups: dict[str, List[SecurityAttackResult]] = defaultdict(list)
        for item in details:
            groups[item.category].append(item)
        by_category = {
            name: SecurityCategorySummary(
                attacks=len(items),
                blocked=sum(item.blocked for item in items),
                block_rate=round(sum(item.blocked for item in items) / len(items), 6),
                side_effects=sum(item.side_effect_detected for item in items),
            )
            for name, items in sorted(groups.items())
        }
        unauthorized_writes = sum(item.side_effect_detected for item in details)
        blocked = sum(item.blocked for item in details)
        return SecurityBenchmarkReport(
            dataset_sha256=security_cases_sha256(cases),
            attacks=len(details),
            blocked=blocked,
            block_rate=round(blocked / len(details), 6),
            unauthorized_writes=unauthorized_writes,
            sandbox_escapes=unauthorized_writes,
            by_category=by_category,
            failures=[
                item for item in details if not item.blocked or item.side_effect_detected
            ],
            details=details,
        )

    def _run_case(
        self, case: SecurityAttackCase, *, repository_root: Path, sentinel: Path
    ) -> SecurityAttackResult:
        sentinel.write_text("safe", encoding="utf-8")
        blocked = False
        scoped_side_effect = False
        error = ""
        try:
            if case.category == "repository_path":
                Repository(repository_root).resolve(case.payload, must_exist=False)
            elif case.category == "scoped_write":
                scoped_side_effect = self._attempt_scoped_write(
                    repository_root, case.payload
                )
            elif case.category == "command_allowlist":
                validate_command(case.payload, {"python", "pytest", "ruff"})
            elif case.category == "snapshot_id":
                validate_snapshot_id(case.payload)
            elif case.category == "archive_path":
                self._attempt_archive_path(repository_root, case.payload)
            elif case.category == "archive_symlink":
                self._attempt_archive_symlink(repository_root, case.payload)
            elif case.category == "docker_policy":
                blocked = self._docker_policy_isolated(repository_root, case.payload)
        except Exception as exc:
            blocked = True
            error = f"{type(exc).__name__}: {exc}"
        side_effect = (
            sentinel.read_text(encoding="utf-8") != "safe" or scoped_side_effect
        )
        return SecurityAttackResult(
            id=case.id,
            category=case.category,
            blocked=blocked and not side_effect,
            side_effect_detected=side_effect,
            error=error,
        )

    @staticmethod
    def _attempt_scoped_write(repository_root: Path, payload: str) -> bool:
        safe_root = repository_root / "src" / "safe"
        safe_root.mkdir(parents=True, exist_ok=True)
        secret = repository_root / "src" / "secret.txt"
        secret.write_text("safe", encoding="utf-8")
        apply_edit(
            ToolContext(
                repository=Repository(repository_root),
                permissions={"write"},
                allowed_paths=["src/safe"],
            ),
            FileEdit(path=payload, old_text="safe", new_text="pwned"),
        )
        return secret.read_text(encoding="utf-8") != "safe"

    @staticmethod
    def _attempt_archive_path(repository_root: Path, payload: str) -> None:
        archive_path = repository_root / "attack.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo(payload)
            content = b"pwned"
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        S3WorkspaceSnapshotStore._extract_archive(archive_path, repository_root / "extract")

    @staticmethod
    def _attempt_archive_symlink(repository_root: Path, payload: str) -> None:
        archive_path = repository_root / "symlink.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo("safe/link")
            info.type = tarfile.SYMTYPE
            info.linkname = payload
            archive.addfile(info)
        S3WorkspaceSnapshotStore._extract_archive(archive_path, repository_root / "extract-link")

    @staticmethod
    def _docker_policy_isolated(repository_root: Path, payload: str) -> bool:
        arguments = DockerSandbox(repository_root).docker_arguments(payload)
        required_pairs = {
            ("--network", "none"),
            ("--security-opt", "no-new-privileges"),
            ("--cap-drop", "ALL"),
            ("--pids-limit", "128"),
        }
        pairs = set(zip(arguments, arguments[1:]))
        return (
            required_pairs.issubset(pairs)
            and "--read-only" in arguments
            and "--memory" in arguments
            and "--cpus" in arguments
            and arguments[-len(validate_command(payload, {"python", "pytest", "ruff"})) :]
            == validate_command(payload, {"python", "pytest", "ruff"})
        )
