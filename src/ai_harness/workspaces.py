from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

SNAPSHOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
IGNORED_WORKSPACE_NAMES = (
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ai-harness",
)


def validate_snapshot_id(snapshot_id: str) -> str:
    if not SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id):
        raise ValueError("invalid workspace snapshot id")
    return snapshot_id


class FilesystemWorkspaceSnapshotStore:
    """Atomic source snapshots on a filesystem shared by API and workers."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def stage(self, snapshot_id: str, source: Path) -> bool:
        snapshot_id = validate_snapshot_id(snapshot_id)
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if self.root == source or self.root.is_relative_to(source):
            raise ValueError("workspace snapshot directory cannot be inside source")
        destination = self.root / snapshot_id
        if destination.is_dir():
            return False
        temporary = self.root / f".{snapshot_id}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copytree(
                source,
                temporary,
                symlinks=True,
                ignore=shutil.ignore_patterns(*IGNORED_WORKSPACE_NAMES),
            )
            try:
                os.rename(temporary, destination)
            except FileExistsError:
                return False
            return True
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def healthcheck(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK | os.X_OK)

    def source_path(self, snapshot_id: str) -> Path:
        snapshot_id = validate_snapshot_id(snapshot_id)
        source = self.root / snapshot_id
        if not source.is_dir():
            raise FileNotFoundError(f"workspace snapshot is unavailable: {snapshot_id}")
        return source

    def remove(self, snapshot_id: str) -> None:
        snapshot_id = validate_snapshot_id(snapshot_id)
        destination = self.root / snapshot_id
        if destination.is_dir():
            shutil.rmtree(destination)

