from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from ai_harness.jobs import (
    CircuitState,
    FailureCircuitBreaker,
    IdempotencyConflictError,
    JobStatus,
    QueueFullError,
    TaskJobStore,
    TaskWorkerPool,
)
from ai_harness.models import TaskRequest, TaskResult, TaskStatus
from ai_harness.orchestrator import ExecutionControl


def request(repository: Path, goal: str = "Fix the queued coding task") -> TaskRequest:
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


def wait_until(predicate, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def test_circuit_breaker_opens_and_allows_single_half_open_probe():
    current_time = [0.0]
    breaker = FailureCircuitBreaker(
        failure_threshold=2,
        recovery_seconds=10,
        clock=lambda: current_time[0],
    )

    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.snapshot().state == CircuitState.CLOSED
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.snapshot().state == CircuitState.OPEN
    assert breaker.allow() is False

    current_time[0] = 10
    assert breaker.allow() is True
    assert breaker.snapshot().state == CircuitState.HALF_OPEN
    assert breaker.allow() is False
    breaker.record_success()

    snapshot = breaker.snapshot()
    assert snapshot.state == CircuitState.CLOSED
    assert snapshot.consecutive_failures == 0


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
    assert [event.event_type for event in reopened.events(queued.job_id)] == [
        "queued",
        "claimed",
        "lease_recovered",
        "completed",
    ]


def test_store_persists_request_id_and_records_it_on_queued_event(tmp_path: Path):
    database = tmp_path / "jobs.sqlite3"
    store = TaskJobStore(database)

    queued = store.enqueue(request(tmp_path), request_id="trace-123")
    reopened = TaskJobStore(database)
    persisted = reopened.get(queued.job_id)

    assert persisted is not None
    assert persisted.request_id == "trace-123"
    assert reopened.events(queued.job_id)[0].details["request_id"] == "trace-123"


def test_store_adds_request_id_column_to_existing_database(tmp_path: Path):
    database = tmp_path / "jobs.sqlite3"
    TaskJobStore(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE task_jobs DROP COLUMN request_id")

    upgraded = TaskJobStore(database)
    queued = upgraded.enqueue(request(tmp_path), request_id="after-upgrade")

    assert upgraded.get(queued.job_id) == queued


def test_expired_lease_records_terminal_failure_when_retry_budget_is_exhausted(
    tmp_path: Path,
):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    queued = store.enqueue(request(tmp_path), max_attempts=1)
    assert store.claim("worker-a", lease_seconds=0.01) is not None

    time.sleep(0.02)
    assert store.claim("worker-b", lease_seconds=1) is None

    failed = store.get(queued.job_id)
    assert failed is not None
    assert failed.status == JobStatus.FAILED
    assert [event.event_type for event in store.events(queued.job_id)] == [
        "queued",
        "claimed",
        "failed",
    ]


def test_idempotency_key_replays_same_job_and_rejects_different_request(
    tmp_path: Path,
):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    original_request = request(tmp_path)

    first = store.enqueue(original_request, idempotency_key="request-123")
    replay = store.enqueue(original_request, idempotency_key="request-123")

    assert replay.job_id == first.job_id
    assert replay.idempotency_key == "request-123"
    assert [event.event_type for event in store.events(first.job_id)] == ["queued"]
    with pytest.raises(IdempotencyConflictError, match="different request"):
        store.enqueue(
            request(tmp_path, "A different coding task"),
            idempotency_key="request-123",
        )


def test_idempotency_keys_are_scoped_per_tenant(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    task_request = request(tmp_path)

    tenant_a = store.enqueue(
        task_request,
        idempotency_key="shared-client-key",
        tenant_id="tenant-a",
        created_by="alice",
    )
    tenant_b = store.enqueue(
        task_request,
        idempotency_key="shared-client-key",
        tenant_id="tenant-b",
        created_by="bob",
    )

    assert tenant_a.job_id != tenant_b.job_id
    assert tenant_a.tenant_id == "tenant-a"
    assert tenant_a.created_by == "alice"
    assert tenant_b.tenant_id == "tenant-b"
    assert tenant_b.created_by == "bob"


def test_concurrent_idempotent_submissions_create_one_job(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    task_request = request(tmp_path)
    barrier = threading.Barrier(6)
    job_ids: list[str] = []
    errors: list[Exception] = []

    def submit() -> None:
        try:
            barrier.wait()
            job_ids.append(
                store.enqueue(task_request, idempotency_key="concurrent-key").job_id
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(job_ids) == 6
    assert len(set(job_ids)) == 1
    assert store.metrics().counts["pending"] == 1
    assert [event.event_type for event in store.events(job_ids[0])] == ["queued"]


def test_only_pending_jobs_can_be_cancelled(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    pending = store.enqueue(request(tmp_path))
    cancelled = store.cancel(pending.job_id)

    assert cancelled is not None
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.cancel_requested is True
    assert [event.event_type for event in store.events(pending.job_id)] == [
        "queued",
        "cancelled",
    ]
    assert store.claim("worker", lease_seconds=1) is None

    running = store.enqueue(request(tmp_path, "Run another coding task"))
    assert store.claim("worker", lease_seconds=1) is not None
    unchanged = store.cancel(running.job_id)
    assert unchanged is not None
    assert unchanged.status == JobStatus.RUNNING
    assert unchanged.cancel_requested is True
    assert store.events(running.job_id)[-1].event_type == "cancellation_requested"


def test_waiting_job_cancellation_cannot_be_overwritten_by_late_approval(
    tmp_path: Path,
) -> None:
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    queued = store.enqueue(request(tmp_path))
    assert store.claim("worker", lease_seconds=30) is not None
    assert store.complete(
        queued.job_id,
        "worker",
        result("task-waiting", TaskStatus.NEEDS_INPUT),
    )

    cancelled = store.cancel(queued.job_id)
    assert cancelled is not None
    assert cancelled.status == JobStatus.CANCELLED
    store.update_for_task(result("task-waiting", TaskStatus.SUCCEEDED))

    persisted = store.get(queued.job_id)
    assert persisted is not None
    assert persisted.status == JobStatus.CANCELLED
    assert [event.event_type for event in store.events(queued.job_id)] == [
        "queued",
        "claimed",
        "completed",
        "cancelled",
    ]


def test_queue_capacity_rejects_excess_outstanding_jobs(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3", max_outstanding_jobs=1)
    first = store.enqueue(request(tmp_path))

    with pytest.raises(QueueFullError, match="capacity 1"):
        store.enqueue(request(tmp_path, "another queued task"))

    store.cancel(first.job_id)
    replacement = store.enqueue(request(tmp_path, "replacement queued task"))
    metrics = store.metrics()

    assert replacement.status == JobStatus.PENDING
    assert metrics.counts["pending"] == 1
    assert metrics.counts["cancelled"] == 1
    assert metrics.outstanding_jobs == 1
    assert metrics.queue_capacity == 1
    assert metrics.cancellation_requests == 1


def test_worker_pool_enforces_maximum_concurrency(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    peak = 0

    class BlockingOrchestrator:
        def run(
            self,
            task_request: TaskRequest,
            *,
            control: ExecutionControl,
        ) -> TaskResult:
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
        runtime = pool.runtime_snapshot()
        assert runtime.ready is True
        assert len(runtime.workers) == 2
        assert sum(worker.completed_jobs for worker in runtime.workers) == 3
        assert runtime.circuit.state == CircuitState.CLOSED
    finally:
        release.set()
        pool.stop()


def test_worker_retries_exception_then_records_success(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    calls = 0

    class FlakyOrchestrator:
        def run(self, _: TaskRequest, *, control: ExecutionControl) -> TaskResult:
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


def test_worker_propagates_trusted_tenant_and_actor_to_execution_control(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    captured: list[tuple[str | None, str | None]] = []

    class CapturingOrchestrator:
        def run(self, _: TaskRequest, *, control: ExecutionControl) -> TaskResult:
            captured.append((control.tenant_id, control.actor))
            return result("tenant-task")

    job = store.enqueue(
        request(tmp_path),
        tenant_id="tenant-a",
        created_by="alice",
    )
    pool = TaskWorkerPool(
        CapturingOrchestrator(),  # type: ignore[arg-type]
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

    assert captured == [("tenant-a", "alice")]


def test_worker_circuit_opens_and_pauses_claiming_after_infrastructure_failure(
    tmp_path: Path,
):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")
    calls = 0

    class BrokenOrchestrator:
        def run(self, _: TaskRequest, *, control: ExecutionControl) -> TaskResult:
            nonlocal calls
            calls += 1
            raise ConnectionError("provider unavailable")

    job = store.enqueue(request(tmp_path), max_attempts=2)
    pool = TaskWorkerPool(
        BrokenOrchestrator(),  # type: ignore[arg-type]
        store,
        max_workers=1,
        poll_interval_seconds=0.01,
        lease_seconds=1,
        circuit_failure_threshold=1,
        circuit_recovery_seconds=60,
    )
    pool.start()
    try:
        wait_until(lambda: pool.runtime_snapshot().circuit.state == CircuitState.OPEN)
        time.sleep(0.05)
        paused = store.get(job.job_id)
        runtime = pool.runtime_snapshot()
    finally:
        pool.stop()

    assert paused is not None
    assert paused.status == JobStatus.PENDING
    assert paused.attempts == 1
    assert calls == 1
    assert runtime.ready is False
    assert runtime.workers[0].failed_jobs == 1


def test_running_job_stops_cooperatively_after_cancellation(tmp_path: Path):
    store = TaskJobStore(tmp_path / "jobs.sqlite3")

    class CooperativeOrchestrator:
        def run(self, _: TaskRequest, *, control: ExecutionControl) -> TaskResult:
            while not control.should_cancel():
                time.sleep(0.01)
            return result("cancelled-task", TaskStatus.CANCELLED)

    job = store.enqueue(request(tmp_path))
    pool = TaskWorkerPool(
        CooperativeOrchestrator(),  # type: ignore[arg-type]
        store,
        max_workers=1,
        poll_interval_seconds=0.01,
        lease_seconds=1,
    )
    pool.start()
    try:
        wait_until(lambda: store.get(job.job_id).status == JobStatus.RUNNING)
        requested = store.cancel(job.job_id)
        assert requested is not None
        assert requested.cancel_requested is True
        wait_until(lambda: store.get(job.job_id).status == JobStatus.CANCELLED)
    finally:
        pool.stop()

    cancelled = store.get(job.job_id)
    assert cancelled is not None
    assert cancelled.result is not None
    assert cancelled.result.status == TaskStatus.CANCELLED
