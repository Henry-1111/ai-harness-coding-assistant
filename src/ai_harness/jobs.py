from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Protocol

from pydantic import BaseModel, Field

from .models import TaskRequest, TaskResult, TaskStatus, utc_now
from .orchestrator import ExecutionControl, HarnessOrchestrator


class QueueFullError(RuntimeError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskJob(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    request: TaskRequest
    task_id: Optional[str] = None
    result: Optional[TaskResult] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = Field(default=2, ge=1, le=5)
    cancel_requested: bool = False
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[str] = None
    created_at: str
    updated_at: str
    idempotency_key: Optional[str] = None
    tenant_id: str = "local"
    created_by: str = "system"


class JobEvent(BaseModel):
    job_id: str
    sequence: int
    event_type: str
    status: JobStatus
    worker_id: Optional[str] = None
    details: dict[str, object] = Field(default_factory=dict)
    created_at: str


class JobMetrics(BaseModel):
    counts: dict[str, int]
    outstanding_jobs: int
    queue_capacity: int
    retried_jobs: int
    cancellation_requests: int
    oldest_pending_age_seconds: float


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitSnapshot(BaseModel):
    state: CircuitState
    consecutive_failures: int
    failure_threshold: int
    recovery_seconds: float
    retry_after_seconds: float


class WorkerSnapshot(BaseModel):
    worker_id: str
    state: str
    current_job_id: Optional[str] = None
    completed_jobs: int = 0
    failed_jobs: int = 0
    last_error: Optional[str] = None
    started_at: str
    last_seen_at: str


class RuntimeSnapshot(BaseModel):
    ready: bool
    queue: JobMetrics
    circuit: CircuitSnapshot
    workers: list[WorkerSnapshot]


class TaskJobStoreProtocol(Protocol):
    """Queue contract shared by SQLite and PostgreSQL implementations."""

    max_outstanding_jobs: int

    def enqueue(
        self,
        request: TaskRequest,
        *,
        max_attempts: int = 2,
        idempotency_key: Optional[str] = None,
        tenant_id: str = "local",
        created_by: str = "system",
        job_id: Optional[str] = None,
    ) -> TaskJob: ...

    def get(self, job_id: str) -> Optional[TaskJob]: ...

    def get_by_task_id(self, task_id: str) -> Optional[TaskJob]: ...

    def events(self, job_id: str) -> list[JobEvent]: ...

    def claim(self, worker_id: str, *, lease_seconds: float) -> Optional[TaskJob]: ...

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: float) -> bool: ...

    def complete(self, job_id: str, worker_id: str, result: TaskResult) -> bool: ...

    def fail(self, job_id: str, worker_id: str, error: str) -> bool: ...

    def cancel(self, job_id: str) -> Optional[TaskJob]: ...

    def is_cancel_requested(self, job_id: str) -> bool: ...

    def metrics(self) -> JobMetrics: ...

    def update_for_task(self, result: TaskResult) -> None: ...


class FailureCircuitBreaker:
    """Thread-safe closed/open/half-open breaker for worker infrastructure failures."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_seconds: float = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= failure_threshold <= 100:
            raise ValueError("failure_threshold must be within 1..100")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if self._opened_at is None:
                    return False
                if self._clock() - self._opened_at < self.recovery_seconds:
                    return False
                self._state = CircuitState.HALF_OPEN
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def release_unused_probe(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._probe_in_flight = False

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._state == CircuitState.HALF_OPEN
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
            self._probe_in_flight = False

    def record_neutral(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
            self._probe_in_flight = False

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            retry_after = 0.0
            if self._state == CircuitState.OPEN and self._opened_at is not None:
                retry_after = max(
                    0.0,
                    self.recovery_seconds - (self._clock() - self._opened_at),
                )
            return CircuitSnapshot(
                state=self._state,
                consecutive_failures=self._consecutive_failures,
                failure_threshold=self.failure_threshold,
                recovery_seconds=self.recovery_seconds,
                retry_after_seconds=round(retry_after, 3),
            )


class TaskJobStore:
    """SQLite queue with atomic claims and expiring worker leases."""

    def __init__(self, database_path: Path, *, max_outstanding_jobs: int = 100) -> None:
        if not 1 <= max_outstanding_jobs <= 100_000:
            raise ValueError("max_outstanding_jobs must be within 1..100000")
        self.database_path = database_path
        self.max_outstanding_jobs = max_outstanding_jobs
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS task_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    task_id TEXT,
                    result_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    cancel_requested INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence),
                    FOREIGN KEY (job_id) REFERENCES task_jobs(job_id)
                );
                CREATE INDEX IF NOT EXISTS idx_task_jobs_claim
                    ON task_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_task_jobs_task_id
                    ON task_jobs(task_id);
                """
            )

            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(task_jobs)").fetchall()
            }
            if "idempotency_key" not in columns:
                connection.execute("ALTER TABLE task_jobs ADD COLUMN idempotency_key TEXT")
            if "request_hash" not in columns:
                connection.execute("ALTER TABLE task_jobs ADD COLUMN request_hash TEXT")
            if "tenant_id" not in columns:
                connection.execute(
                    "ALTER TABLE task_jobs ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'local'"
                )
            if "created_by" not in columns:
                connection.execute(
                    "ALTER TABLE task_jobs ADD COLUMN created_by TEXT NOT NULL DEFAULT 'system'"
                )
            connection.execute("DROP INDEX IF EXISTS idx_task_jobs_idempotency_key")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_task_jobs_idempotency_key
                ON task_jobs(tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL
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
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    "SELECT * FROM task_jobs WHERE tenant_id = ? AND idempotency_key = ?",
                    (tenant_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing["request_hash"] != request_hash:
                        connection.execute("ROLLBACK")
                        raise IdempotencyConflictError(
                            "idempotency key was already used with a different request"
                        )
                    connection.execute("COMMIT")
                    return self._decode(existing)
            outstanding = connection.execute(
                """
                SELECT COUNT(*) AS count FROM task_jobs
                WHERE status IN (?, ?, ?)
                """,
                (
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                    JobStatus.NEEDS_INPUT.value,
                ),
            ).fetchone()
            if int(outstanding["count"]) >= self.max_outstanding_jobs:
                connection.execute("ROLLBACK")
                raise QueueFullError(
                    f"job queue capacity {self.max_outstanding_jobs} has been reached"
                )
            connection.execute(
                """
                INSERT INTO task_jobs (
                    job_id, status, request_json, task_id, result_json, error,
                    attempts, max_attempts, cancel_requested, lease_owner,
                    lease_expires_at, created_at, updated_at,
                    idempotency_key, request_hash, tenant_id, created_by
                ) VALUES (
                    ?, ?, ?, NULL, NULL, NULL, 0, ?, 0, NULL, NULL, ?, ?, ?, ?, ?, ?
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
            connection.execute("COMMIT")
        return job

    def get(self, job_id: str) -> Optional[TaskJob]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._decode(row) if row else None

    def get_by_task_id(self, task_id: str) -> Optional[TaskJob]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_jobs WHERE task_id = ? ORDER BY created_at LIMIT 1",
                (task_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def events(self, job_id: str) -> list[JobEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY sequence",
                (job_id,),
            ).fetchall()
        return [self._decode_event(row) for row in rows]

    def claim(self, worker_id: str, *, lease_seconds: float) -> Optional[TaskJob]:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        now_text = now.isoformat()
        lease_expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exhausted = connection.execute(
                """
                SELECT job_id, attempts FROM task_jobs
                WHERE status = ? AND lease_expires_at <= ? AND attempts >= max_attempts
                """,
                (JobStatus.RUNNING.value, now_text),
            ).fetchall()
            connection.execute(
                """
                UPDATE task_jobs
                SET status = ?, error = COALESCE(error, ?), lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE status = ? AND lease_expires_at <= ? AND attempts >= max_attempts
                """,
                (
                    JobStatus.FAILED.value,
                    "worker lease expired and retry budget was exhausted",
                    now_text,
                    JobStatus.RUNNING.value,
                    now_text,
                ),
            )
            for expired in exhausted:
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
                WHERE cancel_requested = 0 AND attempts < max_attempts
                  AND (status = ? OR (status = ? AND lease_expires_at <= ?))
                ORDER BY created_at, job_id
                LIMIT 1
                """,
                (JobStatus.PENDING.value, JobStatus.RUNNING.value, now_text),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute(
                """
                UPDATE task_jobs
                SET status = ?, attempts = attempts + 1, lease_owner = ?,
                    lease_expires_at = ?, error = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JobStatus.RUNNING.value,
                    worker_id,
                    lease_expires,
                    now_text,
                    row["job_id"],
                ),
            )
            claimed = connection.execute(
                "SELECT * FROM task_jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
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
            connection.execute("COMMIT")
        return self._decode(claimed)

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: float) -> bool:
        expires = (utc_now() + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE task_jobs SET lease_expires_at = ?, updated_at = ?
                WHERE job_id = ? AND status = ? AND lease_owner = ?
                """,
                (expires, utc_now().isoformat(), job_id, JobStatus.RUNNING.value, worker_id),
            )
        return cursor.rowcount == 1

    def complete(self, job_id: str, worker_id: str, result: TaskResult) -> bool:
        status = JobStatus(result.status.value)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE task_jobs
                SET status = ?, task_id = ?, result_json = ?, error = ?,
                    lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE job_id = ? AND status = ? AND lease_owner = ?
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
            connection.execute("COMMIT")
        return cursor.rowcount == 1

    def fail(self, job_id: str, worker_id: str, error: str) -> bool:
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT attempts, max_attempts FROM task_jobs
                WHERE job_id = ? AND status = ? AND lease_owner = ?
                """,
                (job_id, JobStatus.RUNNING.value, worker_id),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return False
            status = (
                JobStatus.PENDING
                if int(row["attempts"]) < int(row["max_attempts"])
                else JobStatus.FAILED
            )
            connection.execute(
                """
                UPDATE task_jobs
                SET status = ?, error = ?, lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE job_id = ?
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
            connection.execute("COMMIT")
        return True

    def cancel(self, job_id: str) -> Optional[TaskJob]:
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, cancel_requested FROM task_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            if row["status"] == JobStatus.PENDING.value:
                connection.execute(
                    """
                    UPDATE task_jobs
                    SET status = ?, cancel_requested = 1, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (JobStatus.CANCELLED.value, now, job_id),
                )
                self._append_event(
                    connection,
                    job_id,
                    "cancelled",
                    JobStatus.CANCELLED,
                    created_at=now,
                )
            elif (
                row["status"] == JobStatus.RUNNING.value
                and not bool(row["cancel_requested"])
            ):
                connection.execute(
                    """
                    UPDATE task_jobs
                    SET cancel_requested = 1, updated_at = ?
                    WHERE job_id = ?
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
                "SELECT * FROM task_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            connection.execute("COMMIT")
        return self._decode(updated)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM task_jobs WHERE job_id = ?", (job_id,)
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
                    SUM(CASE WHEN attempts > 1 THEN 1 ELSE 0 END) AS retried_jobs,
                    SUM(CASE WHEN cancel_requested = 1 THEN 1 ELSE 0 END)
                        AS cancellation_requests,
                    MIN(CASE WHEN status = ? THEN created_at END) AS oldest_pending
                FROM task_jobs
                """,
                (JobStatus.PENDING.value,),
            ).fetchone()
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
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT job_id, status, result_json FROM task_jobs WHERE task_id = ?",
                (result.task_id,),
            ).fetchall()
            connection.execute(
                """
                UPDATE task_jobs
                SET status = ?, result_json = ?, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    status.value,
                    result_json,
                    result.error,
                    utc_now().isoformat(),
                    result.task_id,
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
            connection.execute("COMMIT")

    @staticmethod
    def _decode(row: sqlite3.Row) -> TaskJob:
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
    def _decode_event(row: sqlite3.Row) -> JobEvent:
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
        connection: sqlite3.Connection,
        job_id: str,
        event_type: str,
        status: JobStatus,
        *,
        worker_id: Optional[str] = None,
        details: Optional[dict[str, object]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence "
            "FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO job_events (
                job_id, sequence, event_type, status, worker_id, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
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


class TaskWorkerPool:
    """Bounded background workers for durable coding jobs."""

    def __init__(
        self,
        orchestrator: HarnessOrchestrator,
        store: TaskJobStoreProtocol,
        *,
        max_workers: int = 2,
        poll_interval_seconds: float = 0.2,
        lease_seconds: float = 300,
        task_timeout_seconds: float = 1800,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: float = 30,
        terminal_callback: Optional[Callable[[TaskJob], object]] = None,
    ) -> None:
        if not 1 <= max_workers <= 32:
            raise ValueError("max_workers must be within 1..32")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be positive")
        self.orchestrator = orchestrator
        self.store = store
        self.max_workers = max_workers
        self.poll_interval_seconds = poll_interval_seconds
        self.lease_seconds = lease_seconds
        self.task_timeout_seconds = task_timeout_seconds
        self.circuit = FailureCircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_seconds=circuit_recovery_seconds,
        )
        self.terminal_callback = terminal_callback
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._workers: dict[str, WorkerSnapshot] = {}
        self._worker_lock = threading.Lock()

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for index in range(self.max_workers):
            worker_id = f"worker-{uuid.uuid4()}-{index}"
            now = utc_now().isoformat()
            with self._worker_lock:
                self._workers[worker_id] = WorkerSnapshot(
                    worker_id=worker_id,
                    state="starting",
                    started_at=now,
                    last_seen_at=now,
                )
            thread = threading.Thread(
                target=self._work,
                args=(worker_id,),
                name=f"ai-harness-worker-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout_seconds: float = 5) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout_seconds)
        self._threads.clear()

    def runtime_snapshot(self) -> RuntimeSnapshot:
        circuit = self.circuit.snapshot()
        with self._worker_lock:
            workers = [
                worker.model_copy(deep=True)
                for worker in sorted(self._workers.values(), key=lambda item: item.worker_id)
            ]
        ready = bool(workers) and any(worker.state != "stopped" for worker in workers)
        ready = ready and circuit.state != CircuitState.OPEN
        return RuntimeSnapshot(
            ready=ready,
            queue=self.store.metrics(),
            circuit=circuit,
            workers=workers,
        )

    def _work(self, worker_id: str) -> None:
        self._update_worker(worker_id, state="idle")
        try:
            while not self._stop.is_set():
                if not self.circuit.allow():
                    self._update_worker(worker_id, state="circuit_open")
                    self._stop.wait(self.poll_interval_seconds)
                    continue
                job = self.store.claim(worker_id, lease_seconds=self.lease_seconds)
                if job is None:
                    self.circuit.release_unused_probe()
                    self._update_worker(worker_id, state="idle", current_job_id=None)
                    self._stop.wait(self.poll_interval_seconds)
                    continue
                self._update_worker(
                    worker_id,
                    state="running",
                    current_job_id=job.job_id,
                )
                heartbeat_stop = threading.Event()
                heartbeat = threading.Thread(
                    target=self._heartbeat,
                    args=(job.job_id, worker_id, heartbeat_stop),
                    daemon=True,
                )
                heartbeat.start()
                try:
                    control = ExecutionControl(
                        should_cancel=partial(self.store.is_cancel_requested, job.job_id),
                        deadline=time.monotonic() + self.task_timeout_seconds,
                        tenant_id=job.tenant_id,
                        actor=job.created_by,
                    )
                    result = self.orchestrator.run(job.request, control=control)
                    infrastructure_error = self._infrastructure_error(result)
                    if infrastructure_error:
                        self.circuit.record_failure()
                        self.store.fail(job.job_id, worker_id, infrastructure_error)
                        self._notify_terminal(job.job_id)
                        self._update_worker(
                            worker_id,
                            failed_delta=1,
                            last_error=infrastructure_error,
                        )
                    else:
                        if result.status == TaskStatus.CANCELLED:
                            self.circuit.record_neutral()
                        else:
                            self.circuit.record_success()
                        self.store.complete(job.job_id, worker_id, result)
                        self._notify_terminal(job.job_id)
                        self._update_worker(
                            worker_id,
                            completed_delta=1,
                            failed_delta=int(result.status == TaskStatus.FAILED),
                            last_error=result.error if result.status == TaskStatus.FAILED else None,
                        )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self.circuit.record_failure()
                    self.store.fail(job.job_id, worker_id, error)
                    self._notify_terminal(job.job_id)
                    self._update_worker(
                        worker_id,
                        failed_delta=1,
                        last_error=error,
                    )
                finally:
                    heartbeat_stop.set()
                    heartbeat.join(timeout=1)
                    self._update_worker(worker_id, state="idle", current_job_id=None)
        finally:
            self._update_worker(worker_id, state="stopped", current_job_id=None)

    def _notify_terminal(self, job_id: str) -> None:
        if self.terminal_callback is None:
            return
        job = self.store.get(job_id)
        if job is None or job.status in {
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.NEEDS_INPUT,
        }:
            return
        try:
            self.terminal_callback(job)
        except Exception:
            # The job result is already durable. Quota settlement is idempotent and can be
            # reconciled later without turning a completed coding task into a provider failure.
            return

    @staticmethod
    def _infrastructure_error(result: TaskResult) -> Optional[str]:
        infrastructure_types = {
            "ConnectError",
            "ConnectTimeout",
            "HTTPStatusError",
            "OperationalError",
            "ReadTimeout",
            "RemoteProtocolError",
            "task_timeout",
        }
        for event in reversed(result.trace):
            if event.node != "workflow":
                continue
            error_type = str(event.details.get("error_type", ""))
            if error_type in infrastructure_types:
                return result.error or error_type
            return None
        return None

    def _update_worker(
        self,
        worker_id: str,
        *,
        state: Optional[str] = None,
        current_job_id: object = ...,
        completed_delta: int = 0,
        failed_delta: int = 0,
        last_error: object = ...,
    ) -> None:
        with self._worker_lock:
            worker = self._workers[worker_id]
            if state is not None:
                worker.state = state
            if current_job_id is not ...:
                worker.current_job_id = current_job_id  # type: ignore[assignment]
            if last_error is not ...:
                worker.last_error = last_error  # type: ignore[assignment]
            worker.completed_jobs += completed_delta
            worker.failed_jobs += failed_delta
            worker.last_seen_at = utc_now().isoformat()

    def _heartbeat(
        self,
        job_id: str,
        worker_id: str,
        heartbeat_stop: threading.Event,
    ) -> None:
        interval = max(0.1, self.lease_seconds / 3)
        while not heartbeat_stop.wait(interval):
            if not self.store.heartbeat(
                job_id, worker_id, lease_seconds=self.lease_seconds
            ):
                return
