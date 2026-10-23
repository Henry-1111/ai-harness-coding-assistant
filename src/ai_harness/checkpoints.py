from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Callable, List, Optional, Protocol

from .models import TaskState, utc_now


class CheckpointStoreProtocol(Protocol):
    """Storage contract shared by local and distributed checkpoint backends."""

    def save(self, state: TaskState) -> int: ...

    def load(self, task_id: str) -> Optional[TaskState]: ...

    def load_revision(self, task_id: str, revision: int) -> Optional[TaskState]: ...

    def bisect_first(
        self,
        task_id: str,
        predicate: Callable[[TaskState], bool],
    ) -> Optional[int]: ...

    def revisions(self, task_id: str) -> List[int]: ...


class CheckpointStore:
    """Durable task/session state with append-only revisions."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    current_node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_latest
                    ON task_checkpoints(task_id, revision DESC);
                """
            )

    def save(self, state: TaskState) -> int:
        state.updated_at = utc_now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision "
                "FROM task_checkpoints WHERE task_id = ?",
                (state.task_id,),
            ).fetchone()
            revision = int(row["next_revision"])
            connection.execute(
                "INSERT INTO task_checkpoints "
                "(task_id, revision, current_node, status, state_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    state.task_id,
                    revision,
                    state.current_node,
                    state.status.value,
                    state.model_dump_json(),
                    state.updated_at.isoformat(),
                ),
            )
            return revision

    def load(self, task_id: str) -> Optional[TaskState]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM task_checkpoints WHERE task_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        return TaskState.model_validate_json(row["state_json"]) if row else None

    def load_revision(self, task_id: str, revision: int) -> Optional[TaskState]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM task_checkpoints WHERE task_id = ? AND revision = ?",
                (task_id, revision),
            ).fetchone()
        return TaskState.model_validate_json(row["state_json"]) if row else None

    def bisect_first(
        self,
        task_id: str,
        predicate: Callable[[TaskState], bool],
    ) -> Optional[int]:
        """Find the first revision matching a monotonic predicate in O(log revisions)."""
        revisions = self.revisions(task_id)
        low, high = 0, len(revisions) - 1
        answer: Optional[int] = None
        while low <= high:
            middle = (low + high) // 2
            revision = revisions[middle]
            state = self.load_revision(task_id, revision)
            if state is None:
                return None
            if predicate(state):
                answer = revision
                high = middle - 1
            else:
                low = middle + 1
        return answer

    def revisions(self, task_id: str) -> List[int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT revision FROM task_checkpoints WHERE task_id = ? ORDER BY revision",
                (task_id,),
            ).fetchall()
        return [int(row["revision"]) for row in rows]
