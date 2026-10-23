from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, cast

import psycopg
from psycopg.rows import dict_row

from .memory import MemoryLayer
from .models import utc_now
from .postgres_schema import apply_postgres_migrations


class PostgresLayeredMemoryStore:
    """Shared layered memory with atomic bounded-list updates."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._initialize()

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        return cast(
            psycopg.Connection[dict[str, Any]],
            psycopg.connect(
                self.database_url,
                connect_timeout=5,
                row_factory=cast(Any, dict_row),
            ),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            apply_postgres_migrations(
                connection,
                component="memory",
                migrations=[
                    (
                        1,
                        """
                        CREATE TABLE IF NOT EXISTS memories (
                            layer TEXT NOT NULL,
                            owner_id TEXT NOT NULL,
                            key TEXT NOT NULL,
                            value_json TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            PRIMARY KEY (layer, owner_id, key)
                        )
                        """,
                    )
                ],
            )

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone() is not None

    def put(self, layer: MemoryLayer, owner_id: str, key: str, value: Any) -> None:
        if not owner_id or not key:
            raise ValueError("owner_id and key are required")
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories(layer, owner_id, key, value_json, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(layer, owner_id, key) DO UPDATE SET
                    value_json = EXCLUDED.value_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (layer.value, owner_id, key, encoded, utc_now().isoformat()),
            )

    def get(self, layer: MemoryLayer, owner_id: str, key: str) -> Optional[Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT value_json FROM memories
                WHERE layer = %s AND owner_id = %s AND key = %s
                """,
                (layer.value, owner_id, key),
            ).fetchone()
        return json.loads(row["value_json"]) if row else None

    def append_bounded(
        self,
        layer: MemoryLayer,
        owner_id: str,
        key: str,
        value: Dict[str, Any],
        *,
        max_items: int = 20,
        unique_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not owner_id or not key:
            raise ValueError("owner_id and key are required")
        if max_items < 1:
            raise ValueError("max_items must be positive")
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"memory:{layer.value}:{owner_id}:{key}",),
            )
            row = connection.execute(
                """
                SELECT value_json FROM memories
                WHERE layer = %s AND owner_id = %s AND key = %s
                """,
                (layer.value, owner_id, key),
            ).fetchone()
            existing = json.loads(row["value_json"]) if row else []
            if not isinstance(existing, list):
                raise ValueError("bounded memory value must be a list")
            if unique_by and unique_by in value:
                existing = [
                    item
                    for item in existing
                    if not isinstance(item, dict) or item.get(unique_by) != value.get(unique_by)
                ]
            updated = [*existing, value][-max_items:]
            encoded = json.dumps(updated, ensure_ascii=False, separators=(",", ":"))
            connection.execute(
                """
                INSERT INTO memories(layer, owner_id, key, value_json, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(layer, owner_id, key) DO UPDATE SET
                    value_json = EXCLUDED.value_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (layer.value, owner_id, key, encoded, utc_now().isoformat()),
            )
        return updated

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
