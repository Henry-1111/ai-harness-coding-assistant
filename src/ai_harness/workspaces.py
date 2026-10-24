from __future__ import annotations

import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Iterator

SNAPSHOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
IGNORED_WORKSPACE_NAMES = (
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ai-harness",
)
DEFAULT_SNAPSHOT_CAPACITY_BYTES = 10 * 1024 * 1024 * 1024


class WorkspaceSnapshotCapacityError(RuntimeError):
    """Raised before a snapshot would exceed the configured storage budget."""


@dataclass(frozen=True)
class WorkspaceSnapshotMetrics:
    snapshot_count: int
    bytes_used: int
    max_bytes: int


def validate_snapshot_id(snapshot_id: str) -> str:
    if not SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id):
        raise ValueError("invalid workspace snapshot id")
    return snapshot_id


class FilesystemWorkspaceSnapshotStore:
    """Atomic source snapshots on a filesystem shared by API and workers."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = DEFAULT_SNAPSHOT_CAPACITY_BYTES,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("workspace snapshot capacity must be positive")
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _capacity_lock(self) -> Iterator[None]:
        lock_path = self.root / ".capacity.lock"
        with lock_path.open("a+b") as lock_file:
            flock(lock_file.fileno(), LOCK_EX)
            try:
                yield
            finally:
                flock(lock_file.fileno(), LOCK_UN)

    @staticmethod
    def _tree_size(root: Path, *, exclude_ignored: bool = False) -> int:
        total = 0
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if exclude_ignored and entry.name in IGNORED_WORKSPACE_NAMES:
                        continue
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
        return total

    def _usage_unlocked(self, *, excluding: Path | None = None) -> tuple[int, int]:
        count = 0
        total = 0
        for child in self.root.iterdir():
            if child.name.startswith(".") or child == excluding or not child.is_dir():
                continue
            count += 1
            total += self._tree_size(child)
        return count, total

    def _check_capacity(self, used_bytes: int, candidate_bytes: int) -> None:
        if used_bytes + candidate_bytes > self.max_bytes:
            available = max(0, self.max_bytes - used_bytes)
            raise WorkspaceSnapshotCapacityError(
                "workspace snapshot exceeds storage capacity: "
                f"requires {candidate_bytes} bytes, {available} bytes available"
            )

    def stage(self, snapshot_id: str, source: Path) -> bool:
        snapshot_id = validate_snapshot_id(snapshot_id)
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if self.root == source or self.root.is_relative_to(source):
            raise ValueError("workspace snapshot directory cannot be inside source")
        destination = self.root / snapshot_id
        with self._capacity_lock():
            if destination.is_dir():
                return False
            candidate_bytes = self._tree_size(source, exclude_ignored=True)
            _, used_bytes = self._usage_unlocked()
            self._check_capacity(used_bytes, candidate_bytes)
            temporary = self.root / f".{snapshot_id}.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copytree(
                    source,
                    temporary,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(*IGNORED_WORKSPACE_NAMES),
                )
                actual_bytes = self._tree_size(temporary)
                self._check_capacity(used_bytes, actual_bytes)
                os.rename(temporary, destination)
                return True
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    def replace(self, snapshot_id: str, source: Path) -> None:
        snapshot_id = validate_snapshot_id(snapshot_id)
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = self.root / snapshot_id
        with self._capacity_lock():
            candidate_bytes = self._tree_size(source, exclude_ignored=True)
            _, used_bytes = self._usage_unlocked(excluding=destination)
            self._check_capacity(used_bytes, candidate_bytes)
            temporary = self.root / f".{snapshot_id}.{uuid.uuid4().hex}.tmp"
            backup = self.root / f".{snapshot_id}.{uuid.uuid4().hex}.backup"
            try:
                shutil.copytree(
                    source,
                    temporary,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(*IGNORED_WORKSPACE_NAMES),
                )
                actual_bytes = self._tree_size(temporary)
                self._check_capacity(used_bytes, actual_bytes)
                if destination.exists():
                    os.rename(destination, backup)
                try:
                    os.rename(temporary, destination)
                except Exception:
                    if backup.exists() and not destination.exists():
                        os.rename(backup, destination)
                    raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
                if backup.exists():
                    shutil.rmtree(backup)

    def restore(self, snapshot_id: str, destination: Path) -> Path:
        source = self.source_path(snapshot_id)
        destination = destination.resolve()
        if destination.is_dir():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copytree(source, temporary, symlinks=True)
            try:
                os.rename(temporary, destination)
            except FileExistsError:
                pass
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def healthcheck(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK | os.X_OK)

    def metrics(self) -> WorkspaceSnapshotMetrics:
        with self._capacity_lock():
            snapshot_count, bytes_used = self._usage_unlocked()
        return WorkspaceSnapshotMetrics(
            snapshot_count=snapshot_count,
            bytes_used=bytes_used,
            max_bytes=self.max_bytes,
        )

    def source_path(self, snapshot_id: str) -> Path:
        snapshot_id = validate_snapshot_id(snapshot_id)
        source = self.root / snapshot_id
        if not source.is_dir():
            raise FileNotFoundError(f"workspace snapshot is unavailable: {snapshot_id}")
        return source

    def remove(self, snapshot_id: str) -> None:
        snapshot_id = validate_snapshot_id(snapshot_id)
        destination = self.root / snapshot_id
        with self._capacity_lock():
            if destination.is_dir():
                shutil.rmtree(destination)

