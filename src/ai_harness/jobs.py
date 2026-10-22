from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .models import TaskRequest, TaskResult, utc_now
from .orchestrator import ExecutionControl, HarnessOrchestrator


class QueueFullError(RuntimeError):
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


class JobMetrics(BaseModel):
    counts: dict[str, int]
    outstanding_jobs: int
    queue_capacity: int
    retried_jobs: int
    cancellation_requests: int
    oldest_pending_age_seconds: float


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
                CREATE INDEX IF NOT EXISTS idx_task_jobs_claim
                    ON task_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_task_jobs_task_id
                    ON task_jobs(task_id);
                """
            )

    def enqueue(self, request: TaskRequest, *, max_attempts: int = 2) -> TaskJob:
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be within 1..5")
        now = utc_now().isoformat()
        job = TaskJob(
            job_id=str(uuid.uuid4()),
            request=request,
            max_attempts=max_attempts,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                    lease_expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, NULL, NULL, 0, ?, 0, NULL, NULL, ?, ?)
                """,
                (
                    job.job_id,
                    job.status.value,
                    request.model_dump_json(),
                    max_attempts,
                    now,
                    now,
                ),
            )
            connection.execute("COMMIT")
        return job

    def get(self, job_id: str) -> Optional[TaskJob]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._decode(row) if row else None

    def claim(self, worker_id: str, *, lease_seconds: float) -> Optional[TaskJob]:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        now_text = now.isoformat()
        lease_expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
            row = connection.execute(
                """
                SELECT job_id FROM task_jobs
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
            connection.execute("COMMIT")
        return True

    def cancel(self, job_id: str) -> Optional[TaskJob]:
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM task_jobs WHERE job_id = ?", (job_id,)
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
            elif row["status"] == JobStatus.RUNNING.value:
                connection.execute(
                    """
                    UPDATE task_jobs
                    SET cancel_requested = 1, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (now, job_id),
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
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE task_jobs
                SET status = ?, result_json = ?, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    status.value,
                    result.model_dump_json(),
                    result.error,
                    utc_now().isoformat(),
                    result.task_id,
                ),
            )

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
        )


class TaskWorkerPool:
    """Bounded background workers for durable coding jobs."""

    def __init__(
        self,
        orchestrator: HarnessOrchestrator,
        store: TaskJobStore,
        *,
        max_workers: int = 2,
        poll_interval_seconds: float = 0.2,
        lease_seconds: float = 300,
        task_timeout_seconds: float = 1800,
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
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for index in range(self.max_workers):
            thread = threading.Thread(
                target=self._work,
                args=(f"worker-{uuid.uuid4()}-{index}",),
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

    def _work(self, worker_id: str) -> None:
        while not self._stop.is_set():
            job = self.store.claim(worker_id, lease_seconds=self.lease_seconds)
            if job is None:
                self._stop.wait(self.poll_interval_seconds)
                continue
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
                )
                result = self.orchestrator.run(job.request, control=control)
                self.store.complete(job.job_id, worker_id, result)
            except Exception as exc:
                self.store.fail(job.job_id, worker_id, f"{type(exc).__name__}: {exc}")
            finally:
                heartbeat_stop.set()
                heartbeat.join(timeout=1)

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
