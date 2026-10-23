from __future__ import annotations

from typing import Any, Iterable, Mapping, cast

import psycopg

Migration = tuple[int, str]


def apply_postgres_migrations(
    connection: psycopg.Connection[Any],
    *,
    component: str,
    migrations: Iterable[Migration],
) -> int:
    """Apply ordered component migrations under one cluster-wide transaction lock."""
    ordered = sorted(migrations, key=lambda item: item[0])
    versions = [version for version, _ in ordered]
    if not component.strip():
        raise ValueError("schema component is required")
    if not ordered or versions != list(range(1, len(ordered) + 1)):
        raise ValueError("schema migrations must be contiguous and start at version 1")

    connection.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        ("ai-harness:schema-bootstrap",),
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_harness_schema_migrations (
            component TEXT NOT NULL,
            version INTEGER NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (component, version)
        )
        """
    )
    row = connection.execute(
        """
        SELECT COALESCE(MAX(version), 0) AS version
        FROM ai_harness_schema_migrations WHERE component = %s
        """,
        (component,),
    ).fetchone()
    assert row is not None
    current = int(row["version"] if isinstance(row, Mapping) else row[0])
    latest = versions[-1]
    if current > latest:
        raise RuntimeError(
            f"database schema for {component} is version {current}, "
            f"but this build supports only version {latest}"
        )
    for version, statement in ordered:
        if version <= current:
            continue
        connection.execute(cast(Any, statement))
        connection.execute(
            """
            INSERT INTO ai_harness_schema_migrations (component, version)
            VALUES (%s, %s)
            """,
            (component, version),
        )
        current = version
    return current
