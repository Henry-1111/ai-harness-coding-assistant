from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .governance import (
    APIKeyAuthenticator,
    AuditEvent,
    AuthenticationError,
    AuthorizationError,
    GovernanceStore,
    GovernanceStoreProtocol,
    Principal,
    QuotaExceededError,
    ResourceBinding,
    Role,
    TenantQuotaSnapshot,
)
from .jobs import (
    CircuitState,
    IdempotencyConflictError,
    JobEvent,
    JobMetrics,
    JobStatus,
    QueueFullError,
    RuntimeSnapshot,
    TaskJob,
    TaskJobStore,
    TaskJobStoreProtocol,
    TaskWorkerPool,
)
from .models import TaskRequest, TaskResult, TaskState, TaskStatus
from .observability import HTTPMetrics
from .orchestrator import ExecutionControl, HarnessOrchestrator
from .quota_reconciliation import QuotaReconciler


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    actor: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=1000)


def create_app(
    orchestrator: HarnessOrchestrator,
    *,
    job_store: Optional[TaskJobStoreProtocol] = None,
    worker_pool: Optional[TaskWorkerPool] = None,
    max_job_workers: int = 2,
    job_poll_interval_seconds: float = 0.2,
    job_lease_seconds: float = 300,
    job_queue_capacity: int = 100,
    job_timeout_seconds: float = 1800,
    circuit_failure_threshold: int = 3,
    circuit_recovery_seconds: float = 30,
    authenticator: Optional[APIKeyAuthenticator] = None,
    governance_store: Optional[GovernanceStoreProtocol] = None,
    quota_reconciler: Optional[QuotaReconciler] = None,
    quota_reconciliation_interval_seconds: float = 30,
) -> FastAPI:
    authenticator = authenticator or APIKeyAuthenticator.disabled()
    governance_store = governance_store or GovernanceStore(
        orchestrator.data_directory / "governance.sqlite3"
    )
    job_store = job_store or TaskJobStore(
        orchestrator.data_directory / "jobs.sqlite3",
        max_outstanding_jobs=job_queue_capacity,
    )

    def settle_job_quota(job: TaskJob) -> bool:
        if not authenticator.enabled:
            return False
        actual_tokens = (
            0
            if job.status == JobStatus.CANCELLED and job.attempts == 0
            else job.result.model_usage.total_tokens
            if job.result
            else None
        )
        if governance_store.settle_quota(job.job_id, actual_tokens):
            governance_store.audit(
                actor=job.created_by,
                tenant_id=job.tenant_id,
                action="job.quota.settle",
                resource_type="job",
                resource_id=job.job_id,
                outcome="accepted",
                details={"actual_tokens": actual_tokens},
            )
            return True
        return False

    worker_pool = worker_pool or TaskWorkerPool(
        orchestrator,
        job_store,
        max_workers=max_job_workers,
        poll_interval_seconds=job_poll_interval_seconds,
        lease_seconds=job_lease_seconds,
        task_timeout_seconds=job_timeout_seconds,
        circuit_failure_threshold=circuit_failure_threshold,
        circuit_recovery_seconds=circuit_recovery_seconds,
        terminal_callback=settle_job_quota,
    )
    quota_reconciler = quota_reconciler or QuotaReconciler(
        governance_store,
        job_store,
        settle_job_quota,
        interval_seconds=quota_reconciliation_interval_seconds,
    )
    http_metrics = HTTPMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        worker_pool.start()
        quota_reconciler.start()
        try:
            yield
        finally:
            quota_reconciler.stop()
            worker_pool.stop()

    app = FastAPI(title="AI Harness", version="0.1.0", lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.job_store = job_store
    app.state.worker_pool = worker_pool
    app.state.authenticator = authenticator
    app.state.governance_store = governance_store
    app.state.quota_reconciler = quota_reconciler
    app.state.http_metrics = http_metrics
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )

    @app.middleware("http")
    async def collect_http_metrics(request: Request, call_next) -> Response:
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            route_object = request.scope.get("route")
            route = str(getattr(route_object, "path", "unmatched"))
            http_metrics.observe(
                request.method,
                route,
                500,
                time.perf_counter() - started_at,
            )
            raise
        route_object = request.scope.get("route")
        route = str(getattr(route_object, "path", "unmatched"))
        http_metrics.observe(
            request.method,
            route,
            response.status_code,
            time.perf_counter() - started_at,
        )
        return response

    def current_principal(
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
    ) -> Principal:
        try:
            return authenticator.authenticate(authorization)
        except AuthenticationError as exc:
            governance_store.audit(
                actor="anonymous",
                action="authenticate",
                outcome="denied",
                details={"reason": str(exc)},
            )
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def require_roles(
        principal: Principal,
        *roles: Role,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> None:
        if not principal.authentication_enabled or principal.has_any_role(*roles):
            return
        governance_store.audit(
            actor=principal.subject,
            tenant_id=principal.tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome="denied",
            details={
                "reason": "required role is missing",
                "required_roles": sorted(role.value for role in roles),
            },
        )
        raise HTTPException(status_code=403, detail="required role is missing")

    def authorize_repository(principal: Principal, repository: str, *, action: str) -> None:
        if principal.permits_repository(repository):
            return
        governance_store.audit(
            actor=principal.subject,
            tenant_id=principal.tenant_id,
            action=action,
            resource_type="repository",
            resource_id=str(repository),
            outcome="denied",
            details={"reason": "repository is outside the API key allowlist"},
        )
        raise HTTPException(
            status_code=403,
            detail="repository is outside the API key allowlist",
        )

    def authorize_binding(
        principal: Principal,
        resource_type: str,
        resource_id: str,
        *,
        action: str,
    ) -> ResourceBinding:
        try:
            return governance_store.authorize(principal, resource_type, resource_id)
        except AuthorizationError as exc:
            if not principal.authentication_enabled:
                raise HTTPException(status_code=404, detail="resource not found") from exc
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                outcome="denied",
                details={"reason": str(exc)},
            )
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def authorize_job(
        principal: Principal,
        job: TaskJob,
        *,
        action: str,
    ) -> ResourceBinding:
        binding = governance_store.binding("job", job.job_id)
        if binding is None and not principal.authentication_enabled:
            binding = governance_store.bind(
                "job",
                job.job_id,
                principal,
                repository=job.request.repository,
            )
        if binding is None:
            return authorize_binding(principal, "job", job.job_id, action=action)
        binding = authorize_binding(principal, "job", job.job_id, action=action)
        if job.task_id and governance_store.binding("task", job.task_id) is None:
            governance_store.bind_from("task", job.task_id, binding)
        return binding

    def authorize_task(
        principal: Principal,
        task_id: str,
        *,
        action: str,
    ) -> ResourceBinding:
        binding = governance_store.binding("task", task_id)
        if binding is None:
            job = job_store.get_by_task_id(task_id)
            if job is not None:
                owner = authorize_job(principal, job, action=action)
                binding = governance_store.bind_from("task", task_id, owner)
        if binding is None and not principal.authentication_enabled:
            task = orchestrator.get(task_id)
            if task is not None:
                binding = governance_store.bind(
                    "task",
                    task_id,
                    principal,
                    repository=task.request.repository,
                )
        return authorize_binding(principal, "task", task_id, action=action)

    def dependency_health() -> dict[str, bool]:
        probes = {
            "checkpoints": orchestrator.checkpoints.healthcheck,
            "jobs": job_store.healthcheck,
            "governance": governance_store.healthcheck,
        }
        health: dict[str, bool] = {}
        for name, probe in probes.items():
            try:
                health[name] = bool(probe())
            except Exception:
                health[name] = False
        return health

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/auth/me")
    def auth_identity(
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> dict[str, object]:
        return {
            "key_id": principal.key_id,
            "tenant_id": principal.tenant_id,
            "subject": principal.subject,
            "roles": sorted(role.value for role in principal.roles),
            "authentication_enabled": principal.authentication_enabled,
            "authentication_method": principal.authentication_method,
        }

    @app.get("/ready")
    def readiness() -> dict[str, str]:
        dependencies = dependency_health()
        unavailable = sorted(name for name, healthy in dependencies.items() if not healthy)
        if unavailable:
            raise HTTPException(
                status_code=503,
                detail=f"dependencies unavailable: {', '.join(unavailable)}",
            )
        try:
            runtime = worker_pool.runtime_snapshot()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="runtime storage unavailable") from exc
        if not runtime.ready:
            raise HTTPException(
                status_code=503,
                detail=f"runtime not ready; circuit={runtime.circuit.state.value}",
            )
        return {"status": "ready"}

    @app.get("/v1/runtime", response_model=RuntimeSnapshot)
    def runtime(
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> RuntimeSnapshot:
        require_roles(principal, Role.OPERATOR, action="runtime.read")
        return worker_pool.runtime_snapshot()

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics(
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> str:
        require_roles(principal, Role.OPERATOR, action="metrics.read")
        snapshot = worker_pool.runtime_snapshot()
        reconciliation = quota_reconciler.snapshot()
        dependencies = dependency_health()
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
                "# TYPE ai_harness_quota_reconciliation_runs counter",
                f"ai_harness_quota_reconciliation_runs {reconciliation.runs}",
                "# TYPE ai_harness_quota_reconciliation_settled counter",
                "ai_harness_quota_reconciliation_settled "
                f"{reconciliation.settled_reservations}",
                "# TYPE ai_harness_quota_reconciliation_failures counter",
                f"ai_harness_quota_reconciliation_failures {reconciliation.failures}",
                "# TYPE ai_harness_dependency_ready gauge",
            ]
        )
        lines.extend(
            f'ai_harness_dependency_ready{{dependency="{name}"}} {int(healthy)}'
            for name, healthy in sorted(dependencies.items())
        )
        lines.extend(http_metrics.render_prometheus())
        return "\n".join(lines) + "\n"

    @app.post("/v1/tasks", response_model=TaskResult)
    def create_task(
        request: TaskRequest,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TaskResult:
        require_roles(principal, Role.SUBMITTER, action="task.create")
        authorize_repository(principal, request.repository, action="task.create")
        reservation_id = f"sync-{uuid.uuid4()}"
        if authenticator.enabled:
            try:
                governance_store.reserve_quota(
                    job_id=reservation_id,
                    tenant_id=principal.tenant_id,
                    policy=authenticator.quota_for(principal.tenant_id),
                )
            except QuotaExceededError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action="task.quota.reserve",
                resource_type="quota_reservation",
                resource_id=reservation_id,
                outcome="accepted",
                details={
                    "reserved_tokens": authenticator.quota_for(
                        principal.tenant_id
                    ).token_reservation
                },
            )
        try:
            result = orchestrator.run(
                request,
                control=ExecutionControl(
                    should_cancel=lambda: False,
                    tenant_id=principal.tenant_id,
                    actor=principal.subject,
                ),
            )
        except Exception:
            if authenticator.enabled:
                governance_store.settle_quota(reservation_id, None)
            raise
        if authenticator.enabled:
            actual_tokens = result.model_usage.total_tokens
            governance_store.settle_quota(reservation_id, actual_tokens)
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action="task.quota.settle",
                resource_type="quota_reservation",
                resource_id=reservation_id,
                outcome="accepted",
                details={"actual_tokens": actual_tokens},
            )
        governance_store.bind(
            "task",
            result.task_id,
            principal,
            repository=request.repository,
        )
        governance_store.audit(
            actor=principal.subject,
            tenant_id=principal.tenant_id,
            action="task.create",
            resource_type="task",
            resource_id=result.task_id,
            outcome="accepted",
            details={"repository": str(request.repository)},
        )
        return result

    @app.post("/v1/jobs", response_model=TaskJob, status_code=202)
    def create_job(
        request: TaskRequest,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TaskJob:
        require_roles(principal, Role.SUBMITTER, action="job.create")
        authorize_repository(principal, request.repository, action="job.create")
        job_id = (
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"ai-harness:{principal.tenant_id}:{idempotency_key}",
                )
            )
            if idempotency_key
            else str(uuid.uuid4())
        )
        if authenticator.enabled:
            try:
                governance_store.reserve_quota(
                    job_id=job_id,
                    tenant_id=principal.tenant_id,
                    policy=authenticator.quota_for(principal.tenant_id),
                )
            except QuotaExceededError as exc:
                governance_store.audit(
                    actor=principal.subject,
                    tenant_id=principal.tenant_id,
                    action="job.quota.reserve",
                    resource_type="job",
                    resource_id=job_id,
                    outcome="denied",
                    details={"reason": str(exc)},
                )
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action="job.quota.reserve",
                resource_type="job",
                resource_id=job_id,
                outcome="accepted",
                details={
                    "reserved_tokens": authenticator.quota_for(
                        principal.tenant_id
                    ).token_reservation
                },
            )
        try:
            job = job_store.enqueue(
                request,
                idempotency_key=idempotency_key,
                tenant_id=principal.tenant_id,
                created_by=principal.subject,
                job_id=job_id,
            )
            governance_store.bind(
                "job",
                job.job_id,
                principal,
                repository=request.repository,
            )
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action="job.create",
                resource_type="job",
                resource_id=job.job_id,
                outcome="accepted",
                details={"idempotent_replay": job.attempts > 0},
            )
            return job
        except QueueFullError as exc:
            if authenticator.enabled and job_store.get(job_id) is None:
                governance_store.release_quota(job_id)
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except IdempotencyConflictError as exc:
            if authenticator.enabled and job_store.get(job_id) is None:
                governance_store.release_quota(job_id)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            if authenticator.enabled and job_store.get(job_id) is None:
                governance_store.release_quota(job_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/jobs/metrics", response_model=JobMetrics)
    def job_metrics(
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> JobMetrics:
        require_roles(principal, Role.OPERATOR, action="job.metrics.read")
        return job_store.metrics()

    @app.get("/v1/jobs/{job_id}", response_model=TaskJob)
    def get_job(
        job_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TaskJob:
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        authorize_job(principal, job, action="job.read")
        return job

    @app.get("/v1/jobs/{job_id}/events", response_model=list[JobEvent])
    def get_job_events(
        job_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> list[JobEvent]:
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        authorize_job(principal, job, action="job.events.read")
        return job_store.events(job_id)

    @app.post("/v1/jobs/{job_id}/cancel", response_model=TaskJob)
    def cancel_job(
        job_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TaskJob:
        require_roles(
            principal,
            Role.SUBMITTER,
            Role.OPERATOR,
            action="job.cancel",
            resource_type="job",
            resource_id=job_id,
        )
        existing = job_store.get(job_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="job not found")
        authorize_job(principal, existing, action="job.cancel")
        job = job_store.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status == JobStatus.CANCELLED:
            settle_job_quota(job)
        return job

    @app.get("/v1/tasks/{task_id}", response_model=TaskState)
    def get_task(
        task_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TaskState:
        task = orchestrator.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        authorize_task(principal, task_id, action="task.read")
        return task

    @app.post("/v1/tasks/{task_id}/resume", response_model=TaskResult)
    def resume_task(
        task_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TaskResult:
        require_roles(
            principal,
            Role.SUBMITTER,
            action="task.resume",
            resource_type="task",
            resource_id=task_id,
        )
        authorize_task(principal, task_id, action="task.resume")
        try:
            return orchestrator.resume(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/tasks/{task_id}/approval", response_model=TaskResult)
    def decide_approval(
        task_id: str,
        decision: ApprovalDecisionRequest,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TaskResult:
        require_roles(
            principal,
            Role.REVIEWER,
            action="task.approval.decide",
            resource_type="task",
            resource_id=task_id,
        )
        binding = authorize_task(principal, task_id, action="task.approval.decide")
        if principal.authentication_enabled and decision.actor != principal.subject:
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action="task.approval.decide",
                resource_type="task",
                resource_id=task_id,
                outcome="denied",
                details={"reason": "approval actor must match API identity"},
            )
            raise HTTPException(status_code=403, detail="approval actor must match API identity")
        if principal.authentication_enabled and binding.created_by == principal.subject:
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action="task.approval.decide",
                resource_type="task",
                resource_id=task_id,
                outcome="denied",
                details={"reason": "submitters cannot approve their own task"},
            )
            raise HTTPException(
                status_code=403,
                detail="submitters cannot approve their own task",
            )
        try:
            result = orchestrator.decide_approval(
                task_id,
                approved=decision.approved,
                actor=decision.actor,
                note=decision.note,
            )
            job_store.update_for_task(result)
            job = job_store.get_by_task_id(result.task_id)
            if job is not None and result.status not in {
                TaskStatus.PENDING,
                TaskStatus.RUNNING,
                TaskStatus.NEEDS_INPUT,
            }:
                settle_job_quota(job_store.get(job.job_id) or job)
            governance_store.audit(
                actor=principal.subject,
                tenant_id=principal.tenant_id,
                action="task.approval.decide",
                resource_type="task",
                resource_id=task_id,
                outcome="accepted",
                details={"approved": decision.approved},
            )
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/tasks/{task_id}/diff")
    def task_diff(
        task_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> dict[str, str]:
        authorize_task(principal, task_id, action="task.diff.read")
        try:
            return {"task_id": task_id, "diff": orchestrator.diff(task_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/v1/tasks/{task_id}/trace")
    def task_trace(
        task_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> dict[str, object]:
        task = orchestrator.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        authorize_task(principal, task_id, action="task.trace.read")
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
    def promote_task(
        task_id: str,
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> dict[str, object]:
        require_roles(
            principal,
            Role.PUBLISHER,
            action="task.promote",
            resource_type="task",
            resource_id=task_id,
        )
        authorize_task(principal, task_id, action="task.promote")
        try:
            files = orchestrator.promote(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        governance_store.audit(
            actor=principal.subject,
            tenant_id=principal.tenant_id,
            action="task.promote",
            resource_type="task",
            resource_id=task_id,
            outcome="accepted",
            details={"promoted_files": files},
        )
        return {"task_id": task_id, "promoted_files": files}

    @app.get("/v1/audit/events", response_model=list[AuditEvent])
    def audit_events(
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> list[AuditEvent]:
        require_roles(principal, Role.AUDITOR, Role.OPERATOR, action="audit.read")
        return governance_store.events(principal, limit=limit)

    @app.get("/v1/quota", response_model=TenantQuotaSnapshot)
    def tenant_quota(
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> TenantQuotaSnapshot:
        return governance_store.quota_snapshot(
            principal.tenant_id,
            authenticator.quota_for(principal.tenant_id),
        )

    @app.get("/v1/audit/integrity")
    def audit_integrity(
        principal: Principal = Depends(current_principal),  # noqa: B008
    ) -> dict[str, object]:
        require_roles(principal, Role.AUDITOR, Role.OPERATOR, action="audit.verify")
        valid, event_count = governance_store.verify_integrity(principal)
        return {"valid": valid, "event_count": event_count}

    return app
