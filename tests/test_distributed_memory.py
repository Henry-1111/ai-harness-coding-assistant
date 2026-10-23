from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest

from ai_harness.distributed_memory import PostgresLayeredMemoryStore
from ai_harness.memory import MemoryLayer

POSTGRES_URL = os.getenv("AI_HARNESS_TEST_POSTGRES_URL", "")
requires_postgres = pytest.mark.skipif(
    not POSTGRES_URL, reason="distributed memory PostgreSQL is not configured"
)


@requires_postgres
def test_instances_share_layered_memory_and_atomic_bounded_appends() -> None:
    owner = f"project-{uuid.uuid4().hex}"
    first = PostgresLayeredMemoryStore(POSTGRES_URL)
    second = PostgresLayeredMemoryStore(POSTGRES_URL)
    first.put(MemoryLayer.POLICY, "default", owner, {"policy": "safe"})

    assert second.get(MemoryLayer.POLICY, "default", owner) == {"policy": "safe"}

    def append(index: int) -> None:
        store = first if index % 2 == 0 else second
        store.append_bounded(
            MemoryLayer.PROJECT,
            owner,
            "experiences",
            {"task_id": f"task-{index}"},
            max_items=50,
            unique_by="task_id",
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(append, range(20)))

    experiences = first.get(MemoryLayer.PROJECT, owner, "experiences")
    assert isinstance(experiences, list)
    assert {item["task_id"] for item in experiences} == {
        f"task-{index}" for index in range(20)
    }
    resolved = second.resolve(
        "experiences",
        policy_id="default",
        project_id=owner,
        user_id="user-a",
        session_id="session-a",
    )
    assert resolved == [
        {"layer": "project", "owner_id": owner, "value": experiences}
    ]
    assert first.healthcheck() is True
    with psycopg.connect(POSTGRES_URL) as connection:
        version = connection.execute(
            """
            SELECT MAX(version) FROM ai_harness_schema_migrations
            WHERE component = 'memory'
            """
        ).fetchone()
    assert version is not None and version[0] == 1
