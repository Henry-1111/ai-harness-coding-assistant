from __future__ import annotations

import threading
import time
from pathlib import Path

from ai_harness.jobs import JobStatus, TaskJobStore, TaskWorkerPool
from ai_harness.models import TaskRequest, TaskResult, TaskStatus


def request(repository: Path, goal: str = "Fix the queued coding task") -> TaskRequest:
    return TaskRequest(repository=str(repository), goal=goal)


def result(task_id: str) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.SUCCEEDED,
        changed_files=[],
        verification=[],
    )


def wait_until(predicate, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def test_store_persists_claims_and_recovers_an_expired_lease(tmp_path: Path):
    database = tmp_path / "jobs.sqlite3"
    store = TaskJobStore(database)
    queued = store.enqueue(request(tmp_path), max_attempts=2)

    claimed = store.claim("worker-a", lease_seconds=0.01)
    assert claimed is not None
    assert claimed.job_id == queued.job_id
    assert claimed.status == JobStatus.RUNNING
    assert claimed.attempts == 1

    time.sleep(0.02)
    reopened = TaskJobStore(database)
    recovered = reopened.claim("worker-b", lease_seconds=1)
    assert recovered is not None
    assert recovered.job_id == queued.job_id
    assert recovered.attempts == 2
    assert reopened.complete(recovered.job_id, "worker-b", result("task-1"))

    persisted = reopened.get(queued.job_id)
    assert persisted is not None
    assert persisted.status == JobStatus.SUCCEEDED
    assert persisted.task_id == "task-1"


def test_only_pending_jobs_can_be_cancelled(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    pending = store.enqueue(request(tmp_path))
    cancelled = store.cancel(pending.job_id)

    assert cancelled is not None
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.cancel_requested is True
    assert store.claim("worker", lease_seconds=1) is None

    running = store.enqueue(request(tmp_path, "Run another coding task"))
    assert store.claim("worker", lease_seconds=1) is not None
    unchanged = store.cancel(running.job_id)
    assert unchanged is not None
    assert unchanged.status == JobStatus.RUNNING


def test_worker_pool_enforces_maximum_concurrency(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    peak = 0

    class BlockingOrchestrator:
        def run(self, task_request: TaskRequest) -> TaskResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            release.wait(2)
            with lock:
                active -= 1
            return result(task_request.goal)

    jobs = [store.enqueue(request(tmp_path, f"task-{index}")) for index in range(3)]
    pool = TaskWorkerPool(
        BlockingOrchestrator(),  # type: ignore[arg-type]
        store,
        max_workers=2,
        poll_interval_seconds=0.01,
        lease_seconds=1,
    )
    pool.start()
    try:
        wait_until(
            lambda: sum(
                store.get(job.job_id).status == JobStatus.RUNNING for job in jobs
            )
            == 2
        )
        assert peak == 2
        release.set()
        wait_until(
            lambda: all(store.get(job.job_id).status == JobStatus.SUCCEEDED for job in jobs)
        )
    finally:
        release.set()
        pool.stop()


def test_worker_retries_exception_then_records_success(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    calls = 0

    class FlakyOrchestrator:
        def run(self, _: TaskRequest) -> TaskResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary provider failure")
            return result("recovered-task")

    job = store.enqueue(request(tmp_path), max_attempts=2)
    pool = TaskWorkerPool(
        FlakyOrchestrator(),  # type: ignore[arg-type]
        store,
        max_workers=1,
        poll_interval_seconds=0.01,
        lease_seconds=1,
    )
    pool.start()
    try:
        wait_until(lambda: store.get(job.job_id).status == JobStatus.SUCCEEDED)
    finally:
        pool.stop()

    completed = store.get(job.job_id)
    assert completed is not None
    assert completed.attempts == 2
    assert calls == 2
