from __future__ import annotations

import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from .jobs import JobStatus, TaskJobStore, TaskWorkerPool
from .models import TaskRequest, TaskResult, TaskStatus, utc_now
from .orchestrator import ExecutionControl


class LatencyPercentiles(BaseModel):
    p50: float
    p95: float
    p99: float


class RuntimeBenchmarkReport(BaseModel):
    benchmark: str = "synthetic_job_runtime"
    generated_at: str = Field(default_factory=lambda: utc_now().isoformat())
    submitted_jobs: int
    workers: int
    synthetic_work_ms: float
    duration_ms: float
    jobs_per_second: float
    submission_latency_ms: LatencyPercentiles
    completed_jobs: int
    failed_jobs: int
    idempotency_replays: int
    duplicate_executions: int
    peak_concurrency: int


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def run_runtime_benchmark(
    *,
    jobs: int = 100,
    workers: int = 4,
    synthetic_work_ms: float = 10,
) -> RuntimeBenchmarkReport:
    if not 1 <= jobs <= 10_000:
        raise ValueError("jobs must be within 1..10000")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be within 1..32")
    if not 0 <= synthetic_work_ms <= 10_000:
        raise ValueError("synthetic_work_ms must be within 0..10000")

    lock = threading.Lock()
    active = 0
    peak = 0
    executions: Counter[str] = Counter()

    class SyntheticOrchestrator:
        def run(self, request: TaskRequest, *, control: ExecutionControl) -> TaskResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                executions[request.goal] += 1
            try:
                if synthetic_work_ms:
                    time.sleep(synthetic_work_ms / 1000)
                status = TaskStatus.CANCELLED if control.should_cancel() else TaskStatus.SUCCEEDED
                return TaskResult(
                    task_id=request.goal,
                    status=status,
                    changed_files=[],
                    verification=[],
                )
            finally:
                with lock:
                    active -= 1

    with tempfile.TemporaryDirectory(prefix="ai-harness-runtime-benchmark-") as directory:
        root = Path(directory)
        store = TaskJobStore(root / "jobs.sqlite3", max_outstanding_jobs=jobs + 1)
        submitted = []
        submission_latencies = []
        idempotency_replays = 0
        for index in range(jobs):
            request = TaskRequest(
                repository=str(root),
                goal=f"synthetic-task-{index}",
                verification_commands=[],
            )
            started = time.perf_counter()
            job = store.enqueue(request, idempotency_key=f"benchmark-{index}")
            submission_latencies.append((time.perf_counter() - started) * 1000)
            replay = store.enqueue(request, idempotency_key=f"benchmark-{index}")
            idempotency_replays += int(replay.job_id == job.job_id)
            submitted.append(job)

        pool = TaskWorkerPool(
            SyntheticOrchestrator(),  # type: ignore[arg-type]
            store,
            max_workers=workers,
            poll_interval_seconds=0.001,
            lease_seconds=30,
            task_timeout_seconds=max(1, synthetic_work_ms / 1000 * 10),
        )
        started = time.perf_counter()
        pool.start()
        deadline = time.monotonic() + max(30, jobs * synthetic_work_ms / 1000 * 3)
        try:
            while time.monotonic() < deadline:
                current = [store.get(job.job_id) for job in submitted]
                if all(
                    job is not None
                    and job.status not in {JobStatus.PENDING, JobStatus.RUNNING}
                    for job in current
                ):
                    break
                time.sleep(0.005)
            else:
                raise TimeoutError("runtime benchmark did not finish before its deadline")
        finally:
            pool.stop()
        duration_ms = (time.perf_counter() - started) * 1000
        final = [store.get(job.job_id) for job in submitted]

    completed = sum(job is not None and job.status == JobStatus.SUCCEEDED for job in final)
    failed = sum(job is not None and job.status == JobStatus.FAILED for job in final)
    duplicate_executions = sum(max(0, count - 1) for count in executions.values())
    return RuntimeBenchmarkReport(
        submitted_jobs=jobs,
        workers=workers,
        synthetic_work_ms=synthetic_work_ms,
        duration_ms=round(duration_ms, 3),
        jobs_per_second=round(completed / max(duration_ms / 1000, 0.000001), 3),
        submission_latency_ms=LatencyPercentiles(
            p50=round(percentile(submission_latencies, 0.50), 3),
            p95=round(percentile(submission_latencies, 0.95), 3),
            p99=round(percentile(submission_latencies, 0.99), 3),
        ),
        completed_jobs=completed,
        failed_jobs=failed,
        idempotency_replays=idempotency_replays,
        duplicate_executions=duplicate_executions,
        peak_concurrency=peak,
    )
