from __future__ import annotations

import json
import sqlite3
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import utc_now


class MemoryLayer(str, Enum):
    POLICY = "policy"
    PROJECT = "project"
    USER = "user"
    SESSION = "session"
    SCRATCHPAD = "scratchpad"


class LayeredMemoryStore:
    """Namespaced durable memory without cross-project fallback."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    layer TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (layer, owner_id, key)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def put(self, layer: MemoryLayer, owner_id: str, key: str, value: Any) -> None:
        if not owner_id or not key:
            raise ValueError("owner_id and key are required")
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories(layer, owner_id, key, value_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(layer, owner_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (layer.value, owner_id, key, encoded, utc_now().isoformat()),
            )

    def get(self, layer: MemoryLayer, owner_id: str, key: str) -> Optional[Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM memories WHERE layer = ? AND owner_id = ? AND key = ?",
                (layer.value, owner_id, key),
            ).fetchone()
        return json.loads(row["value_json"]) if row else None

    def resolve(
        self,
        key: str,
        *,
        policy_id: str,
        project_id: str,
        user_id: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        scopes = [
            (MemoryLayer.POLICY, policy_id),
            (MemoryLayer.PROJECT, project_id),
            (MemoryLayer.USER, user_id),
            (MemoryLayer.SESSION, session_id),
            (MemoryLayer.SCRATCHPAD, session_id),
        ]
        resolved = []
        for layer, owner_id in scopes:
            value = self.get(layer, owner_id, key)
            if value is not None:
                resolved.append({"layer": layer.value, "owner_id": owner_id, "value": value})
        return resolved
