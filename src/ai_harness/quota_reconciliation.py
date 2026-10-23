from __future__ import annotations

import threading
from typing import Callable, Optional

from pydantic import BaseModel

from .governance import GovernanceStoreProtocol
from .jobs import JobStatus, TaskJob, TaskJobStoreProtocol
from .models import utc_now


class QuotaReconciliationSnapshot(BaseModel):
    running: bool
    runs: int
    inspected_reservations: int
    settled_reservations: int
    failures: int
    last_error: Optional[str] = None
    last_run_at: Optional[str] = None


class QuotaReconciler:
    """Retries quota settlement from durable terminal jobs after transient failures."""

    def __init__(
        self,
        governance_store: GovernanceStoreProtocol,
        job_store: TaskJobStoreProtocol,
        settle: Callable[[TaskJob], bool],
        *,
        interval_seconds: float = 30,
        batch_size: int = 100,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("quota reconciliation interval must be positive")
        if not 1 <= batch_size <= 10_000:
            raise ValueError("quota reconciliation batch size must be within 1..10000")
        self.governance_store = governance_store
        self.job_store = job_store
        self.settle = settle
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._runs = 0
        self._inspected = 0
        self._settled = 0
        self._failures = 0
        self._last_error: Optional[str] = None
        self._last_run_at: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ai-harness-quota-reconciler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
        self._thread = None

    def reconcile_once(self) -> int:
        inspected = 0
        settled = 0
        failures = 0
        last_error: Optional[str] = None
        try:
            job_ids = self.governance_store.active_quota_job_ids(limit=self.batch_size)
        except Exception as exc:
            failures = 1
            last_error = f"{type(exc).__name__}: {exc}"
            job_ids = []
        for job_id in job_ids:
            inspected += 1
            try:
                job = self.job_store.get(job_id)
                if job is None or job.status in {
                    JobStatus.PENDING,
                    JobStatus.RUNNING,
                    JobStatus.NEEDS_INPUT,
                }:
                    continue
                if self.settle(job):
                    settled += 1
            except Exception as exc:
                failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._runs += 1
            self._inspected += inspected
            self._settled += settled
            self._failures += failures
            self._last_error = last_error
            self._last_run_at = utc_now().isoformat()
        return settled

    def snapshot(self) -> QuotaReconciliationSnapshot:
        with self._lock:
            return QuotaReconciliationSnapshot(
                running=self._thread is not None and self._thread.is_alive(),
                runs=self._runs,
                inspected_reservations=self._inspected,
                settled_reservations=self._settled,
                failures=self._failures,
                last_error=self._last_error,
                last_run_at=self._last_run_at,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            self.reconcile_once()
            self._stop.wait(self.interval_seconds)
