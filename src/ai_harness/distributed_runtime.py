from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional, cast

import psycopg
from psycopg.rows import dict_row

from .jobs import (
    IdempotencyConflictError,
    JobEvent,
    JobMetrics,
    JobStatus,
    QueueFullError,
    TaskJob,
)
from .models import TaskRequest, TaskResult, TaskState, utc_now


class PostgresCheckpointStore:
    """Shared append-only checkpoints with per-task revision serialization."""

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

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone() is not None

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                ("ai-harness:schema-bootstrap",),
            )
            connection.execute(
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
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"checkpoint:{state.task_id}",),
            )
            row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision
                FROM task_checkpoints WHERE task_id = %s
                """,
                (state.task_id,),
            ).fetchone()
            assert row is not None
            revision = int(row["next_revision"])
            connection.execute(
                """
                INSERT INTO task_checkpoints (
                    task_id, revision, current_node, status, state_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
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
                """
                SELECT state_json FROM task_checkpoints
                WHERE task_id = %s ORDER BY revision DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return TaskState.model_validate_json(row["state_json"]) if row else None

    def load_revision(self, task_id: str, revision: int) -> Optional[TaskState]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state_json FROM task_checkpoints
                WHERE task_id = %s AND revision = %s
                """,
                (task_id, revision),
            ).fetchone()
        return TaskState.model_validate_json(row["state_json"]) if row else None

    def revisions(self, task_id: str) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT revision FROM task_checkpoints
                WHERE task_id = %s ORDER BY revision
                """,
                (task_id,),
            ).fetchall()
        return [int(row["revision"]) for row in rows]

    def bisect_first(
        self, task_id: str, predicate: Callable[[TaskState], bool]
    ) -> Optional[int]:
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


class PostgresTaskJobStore:
    """Multi-instance queue using row locks and expiring database leases."""

    def __init__(self, database_url: str, *, max_outstanding_jobs: int = 100) -> None:
        if not 1 <= max_outstanding_jobs <= 100_000:
            raise ValueError("max_outstanding_jobs must be within 1..100000")
        self.database_url = database_url
        self.max_outstanding_jobs = max_outstanding_jobs
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

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone() is not None

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                ("ai-harness:schema-bootstrap",),
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    task_id TEXT,
                    result_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    cancel_requested BOOLEAN NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    idempotency_key TEXT,
                    request_hash TEXT,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    created_by TEXT NOT NULL DEFAULT 'system'
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    job_id TEXT NOT NULL REFERENCES task_jobs(job_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_task_jobs_claim
                    ON task_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_task_jobs_task_id
                    ON task_jobs(task_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_task_jobs_idempotency_key
                    ON task_jobs(tenant_id, idempotency_key)
                    WHERE idempotency_key IS NOT NULL;
                """
            )

    def enqueue(
        self,
        request: TaskRequest,
        *,
        max_attempts: int = 2,
        idempotency_key: Optional[str] = None,
        tenant_id: str = "local",
        created_by: str = "system",
        job_id: Optional[str] = None,
    ) -> TaskJob:
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be within 1..5")
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not 1 <= len(idempotency_key) <= 200:
                raise ValueError("idempotency_key must be within 1..200 characters")
        tenant_id = tenant_id.strip()
        created_by = created_by.strip()
        if not tenant_id or not created_by:
            raise ValueError("tenant_id and created_by are required")
        request_json = request.model_dump_json()
        request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        now = utc_now().isoformat()
        job = TaskJob(
            job_id=job_id or str(uuid.uuid4()),
            request=request,
            max_attempts=max_attempts,
            created_at=now,
            updated_at=now,
            idempotency_key=idempotency_key,
            tenant_id=tenant_id,
            created_by=created_by,
        )
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                ("task-jobs:enqueue",),
            )
            if idempotency_key:
                existing = connection.execute(
                    """
                    SELECT * FROM task_jobs
                    WHERE tenant_id = %s AND idempotency_key = %s
                    """,
                    (tenant_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyConflictError(
                            "idempotency key was already used with a different request"
                        )
                    return self._decode(existing)
            outstanding = connection.execute(
                """
                SELECT COUNT(*) AS count FROM task_jobs
                WHERE status IN (%s, %s, %s)
                """,
                (
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                    JobStatus.NEEDS_INPUT.value,
                ),
            ).fetchone()
            assert outstanding is not None
            if int(outstanding["count"]) >= self.max_outstanding_jobs:
                raise QueueFullError(
                    f"job queue capacity {self.max_outstanding_jobs} has been reached"
                )
            connection.execute(
                """
                INSERT INTO task_jobs (
                    job_id, status, request_json, task_id, result_json, error,
                    attempts, max_attempts, cancel_requested, lease_owner,
                    lease_expires_at, created_at, updated_at, idempotency_key,
                    request_hash, tenant_id, created_by
                ) VALUES (
                    %s, %s, %s, NULL, NULL, NULL, 0, %s, FALSE, NULL, NULL,
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    job.job_id,
                    job.status.value,
                    request_json,
                    max_attempts,
                    now,
                    now,
                    idempotency_key,
                    request_hash,
                    tenant_id,
                    created_by,
                ),
            )
            self._append_event(
                connection,
                job.job_id,
                "queued",
                JobStatus.PENDING,
                details={"max_attempts": max_attempts},
                created_at=now,
            )
        return job

    def get(self, job_id: str) -> Optional[TaskJob]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        return self._decode(row) if row else None

    def get_by_task_id(self, task_id: str) -> Optional[TaskJob]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM task_jobs
                WHERE task_id = %s ORDER BY created_at LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def events(self, job_id: str) -> list[JobEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job_events WHERE job_id = %s ORDER BY sequence", (job_id,)
            ).fetchall()
        return [self._decode_event(row) for row in rows]

    def claim(self, worker_id: str, *, lease_seconds: float) -> Optional[TaskJob]:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        now_text = now.isoformat()
        lease_expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            exhausted = connection.execute(
                """
                SELECT job_id, attempts FROM task_jobs
                WHERE status = %s AND lease_expires_at <= %s AND attempts >= max_attempts
                FOR UPDATE SKIP LOCKED
                """,
                (JobStatus.RUNNING.value, now_text),
            ).fetchall()
            for expired in exhausted:
                connection.execute(
                    """
                    UPDATE task_jobs
                    SET status = %s, error = COALESCE(error, %s), lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = %s
                    WHERE job_id = %s
                    """,
                    (
                        JobStatus.FAILED.value,
                        "worker lease expired and retry budget was exhausted",
                        now_text,
                        expired["job_id"],
                    ),
                )
                self._append_event(
                    connection,
                    expired["job_id"],
                    "failed",
                    JobStatus.FAILED,
                    details={
                        "error": "worker lease expired and retry budget was exhausted",
                        "attempt": int(expired["attempts"]),
                    },
                    created_at=now_text,
                )
            row = connection.execute(
                """
                SELECT job_id, status FROM task_jobs
                WHERE cancel_requested = FALSE AND attempts < max_attempts
                  AND (status = %s OR (status = %s AND lease_expires_at <= %s))
                ORDER BY created_at, job_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (JobStatus.PENDING.value, JobStatus.RUNNING.value, now_text),
            ).fetchone()
            if row is None:
                return None
            claimed = connection.execute(
                """
                UPDATE task_jobs
                SET status = %s, attempts = attempts + 1, lease_owner = %s,
                    lease_expires_at = %s, error = NULL, updated_at = %s
                WHERE job_id = %s
                RETURNING *
                """,
                (
                    JobStatus.RUNNING.value,
                    worker_id,
                    lease_expires,
                    now_text,
                    row["job_id"],
                ),
            ).fetchone()
            assert claimed is not None
            self._append_event(
                connection,
                row["job_id"],
                "lease_recovered"
                if row["status"] == JobStatus.RUNNING.value
                else "claimed",
                JobStatus.RUNNING,
                worker_id=worker_id,
                details={"attempt": int(claimed["attempts"])},
                created_at=now_text,
            )
            return self._decode(claimed)

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: float) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE task_jobs SET lease_expires_at = %s, updated_at = %s
                WHERE job_id = %s AND status = %s AND lease_owner = %s
                """,
                (expires, now.isoformat(), job_id, JobStatus.RUNNING.value, worker_id),
            )
            return cursor.rowcount == 1

    def complete(self, job_id: str, worker_id: str, result: TaskResult) -> bool:
        status = JobStatus(result.status.value)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT job_id FROM task_jobs WHERE job_id = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if row is None:
                return False
            cursor = connection.execute(
                """
                UPDATE task_jobs
                SET status = %s, task_id = %s, result_json = %s, error = %s,
                    lease_owner = NULL, lease_expires_at = NULL, updated_at = %s
                WHERE job_id = %s AND status = %s AND lease_owner = %s
                """,
                (
                    status.value,
                    result.task_id,
                    result.model_dump_json(),
                    result.error,
                    utc_now().isoformat(),
                    job_id,
                    JobStatus.RUNNING.value,
                    worker_id,
                ),
            )
            if cursor.rowcount == 1:
                self._append_event(
                    connection,
                    job_id,
                    "completed",
                    status,
                    worker_id=worker_id,
                    details={"task_id": result.task_id},
                )
            return cursor.rowcount == 1

    def fail(self, job_id: str, worker_id: str, error: str) -> bool:
        now = utc_now().isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attempts, max_attempts FROM task_jobs
                WHERE job_id = %s AND status = %s AND lease_owner = %s
                FOR UPDATE
                """,
                (job_id, JobStatus.RUNNING.value, worker_id),
            ).fetchone()
            if row is None:
                return False
            status = (
                JobStatus.PENDING
                if int(row["attempts"]) < int(row["max_attempts"])
                else JobStatus.FAILED
            )
            connection.execute(
                """
                UPDATE task_jobs
                SET status = %s, error = %s, lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = %s
                WHERE job_id = %s
                """,
                (status.value, error[:4000], now, job_id),
            )
            self._append_event(
                connection,
                job_id,
                "retry_scheduled" if status == JobStatus.PENDING else "failed",
                status,
                worker_id=worker_id,
                details={"error": error[:1000], "attempt": int(row["attempts"])},
                created_at=now,
            )
            return True

    def cancel(self, job_id: str) -> Optional[TaskJob]:
        now = utc_now().isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, cancel_requested FROM task_jobs
                WHERE job_id = %s FOR UPDATE
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            if row["status"] in {
                JobStatus.PENDING.value,
                JobStatus.NEEDS_INPUT.value,
            }:
                connection.execute(
                    """
                    UPDATE task_jobs SET status = %s, cancel_requested = TRUE,
                        lease_owner = NULL, lease_expires_at = NULL, updated_at = %s
                    WHERE job_id = %s
                    """,
                    (JobStatus.CANCELLED.value, now, job_id),
                )
                self._append_event(
                    connection, job_id, "cancelled", JobStatus.CANCELLED, created_at=now
                )
            elif row["status"] == JobStatus.RUNNING.value and not bool(
                row["cancel_requested"]
            ):
                connection.execute(
                    """
                    UPDATE task_jobs SET cancel_requested = TRUE, updated_at = %s
                    WHERE job_id = %s
                    """,
                    (now, job_id),
                )
                self._append_event(
                    connection,
                    job_id,
                    "cancellation_requested",
                    JobStatus.RUNNING,
                    created_at=now,
                )
            updated = connection.execute(
                "SELECT * FROM task_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
            assert updated is not None
            return self._decode(updated)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM task_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def metrics(self) -> JobMetrics:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM task_jobs GROUP BY status"
            ).fetchall()
            aggregates = connection.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE attempts > 1) AS retried_jobs,
                    COUNT(*) FILTER (WHERE cancel_requested = TRUE)
                        AS cancellation_requests,
                    MIN(created_at) FILTER (WHERE status = %s) AS oldest_pending
                FROM task_jobs
                """,
                (JobStatus.PENDING.value,),
            ).fetchone()
            assert aggregates is not None
        counts = {status.value: 0 for status in JobStatus}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        outstanding = sum(
            counts[status.value]
            for status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_INPUT)
        )
        oldest_pending_age = 0.0
        if aggregates["oldest_pending"]:
            created_at = datetime.fromisoformat(str(aggregates["oldest_pending"]))
            oldest_pending_age = max(0.0, (utc_now() - created_at).total_seconds())
        return JobMetrics(
            counts=counts,
            outstanding_jobs=outstanding,
            queue_capacity=self.max_outstanding_jobs,
            retried_jobs=int(aggregates["retried_jobs"] or 0),
            cancellation_requests=int(aggregates["cancellation_requests"] or 0),
            oldest_pending_age_seconds=round(oldest_pending_age, 3),
        )

    def update_for_task(self, result: TaskResult) -> None:
        status = JobStatus(result.status.value)
        result_json = result.model_dump_json()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, status, result_json FROM task_jobs
                WHERE task_id = %s AND status = %s FOR UPDATE
                """,
                (result.task_id, JobStatus.NEEDS_INPUT.value),
            ).fetchall()
            connection.execute(
                """
                UPDATE task_jobs
                SET status = %s, result_json = %s, error = %s, updated_at = %s
                WHERE task_id = %s AND status = %s
                """,
                (
                    status.value,
                    result_json,
                    result.error,
                    utc_now().isoformat(),
                    result.task_id,
                    JobStatus.NEEDS_INPUT.value,
                ),
            )
            for row in rows:
                if row["status"] == status.value and row["result_json"] == result_json:
                    continue
                self._append_event(
                    connection,
                    row["job_id"],
                    "approval_completed",
                    status,
                    details={"task_id": result.task_id},
                )

    @staticmethod
    def _decode(row: Mapping[str, Any]) -> TaskJob:
        return TaskJob(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            request=TaskRequest.model_validate_json(row["request_json"]),
            task_id=row["task_id"],
            result=(
                TaskResult.model_validate_json(row["result_json"])
                if row["result_json"]
                else None
            ),
            error=row["error"],
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            cancel_requested=bool(row["cancel_requested"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            idempotency_key=row["idempotency_key"],
            tenant_id=row["tenant_id"],
            created_by=row["created_by"],
        )

    @staticmethod
    def _decode_event(row: Mapping[str, Any]) -> JobEvent:
        return JobEvent(
            job_id=row["job_id"],
            sequence=int(row["sequence"]),
            event_type=row["event_type"],
            status=JobStatus(row["status"]),
            worker_id=row["worker_id"],
            details=json.loads(row["details_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _append_event(
        connection: psycopg.Connection[dict[str, Any]],
        job_id: str,
        event_type: str,
        status: JobStatus,
        *,
        worker_id: Optional[str] = None,
        details: Optional[dict[str, object]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence
            FROM job_events WHERE job_id = %s
            """,
            (job_id,),
        ).fetchone()
        assert row is not None
        connection.execute(
            """
            INSERT INTO job_events (
                job_id, sequence, event_type, status, worker_id, details_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_id,
                int(row["sequence"]),
                event_type,
                status.value,
                worker_id,
                json.dumps(details or {}, ensure_ascii=False, separators=(",", ":")),
                created_at or utc_now().isoformat(),
            ),
        )
