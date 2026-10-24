from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest

from ai_harness.distributed_runtime import PostgresCheckpointStore, PostgresTaskJobStore
from ai_harness.jobs import JobStatus, QueueFullError
from ai_harness.models import TaskRequest, TaskResult, TaskState, TaskStatus

POSTGRES_URL = os.getenv("AI_HARNESS_TEST_POSTGRES_URL", "")
requires_postgres = pytest.mark.skipif(
    not POSTGRES_URL, reason="distributed runtime PostgreSQL is not configured"
)


@requires_postgres
def test_concurrent_schema_bootstrap_is_serialized() -> None:
    with psycopg.connect(POSTGRES_URL) as connection:
        connection.execute(
            """
            DROP TABLE IF EXISTS task_checkpoints, job_events, task_jobs,
                ai_harness_schema_migrations CASCADE
            """
        )

    def initialize(index: int) -> bool:
        store = (
            PostgresCheckpointStore(POSTGRES_URL)
            if index % 2 == 0
            else PostgresTaskJobStore(POSTGRES_URL)
        )
        return store.healthcheck()

    with ThreadPoolExecutor(max_workers=12) as executor:
        health = list(executor.map(initialize, range(12)))

    assert all(health)
    with psycopg.connect(POSTGRES_URL) as connection:
        versions = dict(
            connection.execute(
                """
                SELECT component, MAX(version) FROM ai_harness_schema_migrations
                WHERE component IN ('checkpoints', 'jobs') GROUP BY component
                """
            ).fetchall()
        )
    assert versions == {"checkpoints": 1, "jobs": 2}
    with psycopg.connect(POSTGRES_URL) as connection:
        connection.execute(
            """
            INSERT INTO ai_harness_schema_migrations (component, version)
            VALUES ('checkpoints', 2)
            """
        )
    try:
        with pytest.raises(RuntimeError, match="supports only version 1"):
            PostgresCheckpointStore(POSTGRES_URL)
    finally:
        with psycopg.connect(POSTGRES_URL) as connection:
            connection.execute(
                """
                DELETE FROM ai_harness_schema_migrations
                WHERE component = 'checkpoints' AND version = 2
                """
            )


def reset_runtime_tables() -> None:
    with psycopg.connect(POSTGRES_URL) as connection:
        connection.execute(
            "TRUNCATE task_checkpoints, job_events, task_jobs RESTART IDENTITY CASCADE"
        )


@requires_postgres
def test_jobs_schema_upgrades_from_version_one() -> None:
    PostgresTaskJobStore(POSTGRES_URL)
    with psycopg.connect(POSTGRES_URL) as connection:
        connection.execute("ALTER TABLE task_jobs DROP COLUMN IF EXISTS request_id")
        connection.execute(
            """
            DELETE FROM ai_harness_schema_migrations
            WHERE component = 'jobs' AND version = 2
            """
        )

    upgraded = PostgresTaskJobStore(POSTGRES_URL)
    with psycopg.connect(POSTGRES_URL) as connection:
        columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'task_jobs'
                """
            ).fetchall()
        }
        version = connection.execute(
            """
            SELECT MAX(version) FROM ai_harness_schema_migrations
            WHERE component = 'jobs'
            """
        ).fetchone()

    assert upgraded.healthcheck()
    assert "request_id" in columns
    assert version == (2,)


def request(repository: Path, goal: str) -> TaskRequest:
    return TaskRequest(repository=str(repository), goal=goal)


def result(
    task_id: str, status: TaskStatus = TaskStatus.SUCCEEDED
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=status,
        changed_files=[],
        verification=[],
    )


@requires_postgres
def test_concurrent_idempotency_and_queue_capacity_are_global(tmp_path: Path) -> None:
    reset_runtime_tables()
    first = PostgresTaskJobStore(POSTGRES_URL, max_outstanding_jobs=1)
    second = PostgresTaskJobStore(POSTGRES_URL, max_outstanding_jobs=1)
    task_request = request(tmp_path, "same task")

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = list(
            executor.map(
                lambda index: (first if index % 2 == 0 else second).enqueue(
                    task_request,
                    idempotency_key="same-key",
                    tenant_id="tenant-a",
                    created_by="alice",
                ),
                range(8),
            )
        )

    assert len({job.job_id for job in jobs}) == 1
    assert first.metrics().outstanding_jobs == 1
    with pytest.raises(QueueFullError, match="capacity 1"):
        second.enqueue(request(tmp_path, "another task"))


@requires_postgres
def test_request_id_survives_postgres_round_trip(tmp_path: Path) -> None:
    reset_runtime_tables()
    store = PostgresTaskJobStore(POSTGRES_URL)

    queued = store.enqueue(request(tmp_path, "trace this task"), request_id="trace-123")
    persisted = store.get(queued.job_id)

    assert persisted is not None
    assert persisted.request_id == "trace-123"
    assert store.events(queued.job_id)[0].details["request_id"] == "trace-123"


@requires_postgres
def test_skip_locked_claims_each_job_once_across_instances(tmp_path: Path) -> None:
    reset_runtime_tables()
    first = PostgresTaskJobStore(POSTGRES_URL, max_outstanding_jobs=100)
    second = PostgresTaskJobStore(POSTGRES_URL, max_outstanding_jobs=100)
    for index in range(20):
        first.enqueue(request(tmp_path, f"task-{index}"))

    def claim(index: int) -> str | None:
        store = first if index % 2 == 0 else second
        job = store.claim(f"worker-{index}", lease_seconds=30)
        return job.job_id if job else None

    with ThreadPoolExecutor(max_workers=20) as executor:
        claimed = list(executor.map(claim, range(20)))

    assert None not in claimed
    assert len(set(claimed)) == 20
    assert first.metrics().counts[JobStatus.RUNNING.value] == 20


@requires_postgres
def test_expired_lease_is_recovered_and_stale_worker_cannot_complete(
    tmp_path: Path,
) -> None:
    reset_runtime_tables()
    first = PostgresTaskJobStore(POSTGRES_URL)
    second = PostgresTaskJobStore(POSTGRES_URL)
    queued = first.enqueue(request(tmp_path, "recover lease"), max_attempts=2)
    claimed = first.claim("worker-a", lease_seconds=0.01)
    assert claimed is not None

    time.sleep(0.02)
    recovered = second.claim("worker-b", lease_seconds=30)

    assert recovered is not None
    assert recovered.job_id == queued.job_id
    assert recovered.attempts == 2
    assert first.complete(queued.job_id, "worker-a", result("stale")) is False
    assert second.complete(queued.job_id, "worker-b", result("winner")) is True
    persisted = first.get(queued.job_id)
    assert persisted is not None
    assert persisted.task_id == "winner"
    assert [event.event_type for event in first.events(queued.job_id)] == [
        "queued",
        "claimed",
        "lease_recovered",
        "completed",
    ]


@requires_postgres
def test_cancelled_waiting_job_is_not_revived_by_approval_update(tmp_path: Path) -> None:
    reset_runtime_tables()
    store = PostgresTaskJobStore(POSTGRES_URL)
    queued = store.enqueue(request(tmp_path, "cancel approval wait"))
    assert store.claim("worker-a", lease_seconds=30) is not None
    assert store.complete(
        queued.job_id,
        "worker-a",
        result("task-waiting", TaskStatus.NEEDS_INPUT),
    )

    cancelled = store.cancel(queued.job_id)
    assert cancelled is not None
    assert cancelled.status == JobStatus.CANCELLED
    store.update_for_task(result("task-waiting"))

    persisted = store.get(queued.job_id)
    assert persisted is not None
    assert persisted.status == JobStatus.CANCELLED


@requires_postgres
def test_concurrent_checkpoint_revisions_are_contiguous(tmp_path: Path) -> None:
    reset_runtime_tables()
    first = PostgresCheckpointStore(POSTGRES_URL)
    second = PostgresCheckpointStore(POSTGRES_URL)
    base = TaskState(
        task_id=f"task-{uuid.uuid4().hex}",
        request=request(tmp_path, "checkpoint concurrently"),
    )

    def save(index: int) -> int:
        state = base.model_copy(deep=True)
        state.current_node = f"node-{index}"
        return (first if index % 2 == 0 else second).save(state)

    with ThreadPoolExecutor(max_workers=10) as executor:
        revisions = list(executor.map(save, range(20)))

    assert sorted(revisions) == list(range(1, 21))
    assert first.revisions(base.task_id) == list(range(1, 21))
    assert second.load(base.task_id) is not None
