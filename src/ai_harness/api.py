from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .jobs import (
    CircuitState,
    JobMetrics,
    QueueFullError,
    RuntimeSnapshot,
    TaskJob,
    TaskJobStore,
    TaskWorkerPool,
)
from .models import TaskRequest, TaskResult, TaskState
from .orchestrator import HarnessOrchestrator


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    actor: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=1000)


def create_app(
    orchestrator: HarnessOrchestrator,
    *,
    job_store: Optional[TaskJobStore] = None,
    worker_pool: Optional[TaskWorkerPool] = None,
    max_job_workers: int = 2,
    job_poll_interval_seconds: float = 0.2,
    job_lease_seconds: float = 300,
    job_queue_capacity: int = 100,
    job_timeout_seconds: float = 1800,
    circuit_failure_threshold: int = 3,
    circuit_recovery_seconds: float = 30,
) -> FastAPI:
    job_store = job_store or TaskJobStore(
        orchestrator.data_directory / "jobs.sqlite3",
        max_outstanding_jobs=job_queue_capacity,
    )
    worker_pool = worker_pool or TaskWorkerPool(
        orchestrator,
        job_store,
        max_workers=max_job_workers,
        poll_interval_seconds=job_poll_interval_seconds,
        lease_seconds=job_lease_seconds,
        task_timeout_seconds=job_timeout_seconds,
        circuit_failure_threshold=circuit_failure_threshold,
        circuit_recovery_seconds=circuit_recovery_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        worker_pool.start()
        try:
            yield
        finally:
            worker_pool.stop()

    app = FastAPI(title="AI Harness", version="0.1.0", lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.job_store = job_store
    app.state.worker_pool = worker_pool
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def readiness() -> dict[str, str]:
        runtime = worker_pool.runtime_snapshot()
        if not runtime.ready:
            raise HTTPException(
                status_code=503,
                detail=f"runtime not ready; circuit={runtime.circuit.state.value}",
            )
        return {"status": "ready"}

    @app.get("/v1/runtime", response_model=RuntimeSnapshot)
    def runtime() -> RuntimeSnapshot:
        return worker_pool.runtime_snapshot()

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        snapshot = worker_pool.runtime_snapshot()
        lines = [
            "# HELP ai_harness_ready Whether the worker runtime is ready.",
            "# TYPE ai_harness_ready gauge",
            f"ai_harness_ready {int(snapshot.ready)}",
            "# HELP ai_harness_jobs Number of jobs by status.",
            "# TYPE ai_harness_jobs gauge",
        ]
        lines.extend(
            f'ai_harness_jobs{{status="{status}"}} {count}'
            for status, count in sorted(snapshot.queue.counts.items())
        )
        lines.extend(
            [
                "# TYPE ai_harness_queue_outstanding gauge",
                f"ai_harness_queue_outstanding {snapshot.queue.outstanding_jobs}",
                "# TYPE ai_harness_queue_capacity gauge",
                f"ai_harness_queue_capacity {snapshot.queue.queue_capacity}",
                "# TYPE ai_harness_queue_oldest_pending_seconds gauge",
                "ai_harness_queue_oldest_pending_seconds "
                f"{snapshot.queue.oldest_pending_age_seconds}",
                "# TYPE ai_harness_workers gauge",
                f"ai_harness_workers {len(snapshot.workers)}",
                "# TYPE ai_harness_workers_running gauge",
                "ai_harness_workers_running "
                f"{sum(worker.state == 'running' for worker in snapshot.workers)}",
                "# TYPE ai_harness_circuit_state gauge",
            ]
        )
        lines.extend(
            f'ai_harness_circuit_state{{state="{state.value}"}} '
            f"{int(snapshot.circuit.state == state)}"
            for state in CircuitState
        )
        lines.extend(
            [
                "# TYPE ai_harness_circuit_consecutive_failures gauge",
                "ai_harness_circuit_consecutive_failures "
                f"{snapshot.circuit.consecutive_failures}",
            ]
        )
        return "\n".join(lines) + "\n"

    @app.post("/v1/tasks", response_model=TaskResult)
    def create_task(request: TaskRequest) -> TaskResult:
        return orchestrator.run(request)

    @app.post("/v1/jobs", response_model=TaskJob, status_code=202)
    def create_job(request: TaskRequest) -> TaskJob:
        try:
            return job_store.enqueue(request)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/v1/jobs/metrics", response_model=JobMetrics)
    def job_metrics() -> JobMetrics:
        return job_store.metrics()

    @app.get("/v1/jobs/{job_id}", response_model=TaskJob)
    def get_job(job_id: str) -> TaskJob:
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/v1/jobs/{job_id}/cancel", response_model=TaskJob)
    def cancel_job(job_id: str) -> TaskJob:
        job = job_store.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/v1/tasks/{task_id}", response_model=TaskState)
    def get_task(task_id: str) -> TaskState:
        task = orchestrator.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.post("/v1/tasks/{task_id}/resume", response_model=TaskResult)
    def resume_task(task_id: str) -> TaskResult:
        try:
            return orchestrator.resume(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/tasks/{task_id}/approval", response_model=TaskResult)
    def decide_approval(task_id: str, decision: ApprovalDecisionRequest) -> TaskResult:
        try:
            result = orchestrator.decide_approval(
                task_id,
                approved=decision.approved,
                actor=decision.actor,
                note=decision.note,
            )
            job_store.update_for_task(result)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/tasks/{task_id}/diff")
    def task_diff(task_id: str) -> dict[str, str]:
        try:
            return {"task_id": task_id, "diff": orchestrator.diff(task_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/v1/tasks/{task_id}/trace")
    def task_trace(task_id: str) -> dict[str, object]:
        task = orchestrator.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "task_id": task_id,
            "events": [event.model_dump(mode="json") for event in task.trace],
            "retrieval": (
                task.retrieval_trace.model_dump(mode="json") if task.retrieval_trace else None
            ),
            "restricted_react": task.scratchpad.get("restricted_react"),
            "repairs": [item.model_dump(mode="json") for item in task.repair_history],
            "collaboration": [
                item.model_dump(mode="json") for item in task.collaboration_history
            ],
            "approval": task.approval.model_dump(mode="json") if task.approval else None,
            "checkpoint_revisions": orchestrator.checkpoints.revisions(task_id),
        }

    @app.post("/v1/tasks/{task_id}/promote")
    def promote_task(task_id: str) -> dict[str, object]:
        try:
            files = orchestrator.promote(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"task_id": task_id, "promoted_files": files}

    return app
