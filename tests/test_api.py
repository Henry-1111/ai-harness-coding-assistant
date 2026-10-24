import hashlib
import shutil
import time
from pathlib import Path

from conftest import ScriptedModel
from fastapi.testclient import TestClient

from ai_harness.api import create_app
from ai_harness.governance import (
    APIKeyAuthenticator,
    APIKeyDefinition,
    GovernanceConfig,
    GovernanceStore,
    Role,
    TenantQuotaPolicy,
)
from ai_harness.jobs import TaskJobStore, TaskWorkerPool
from ai_harness.orchestrator import HarnessOrchestrator
from ai_harness.sandbox import LocalCommandSandbox
from ai_harness.workspaces import FilesystemWorkspaceSnapshotStore


def test_api_runs_and_reads_task(tmp_path: Path, sample_repository: Path):
    model = ScriptedModel(
        [
            {
                "summary": "Fix add",
                "edits": [
                    {
                        "path": "calculator.py",
                        "old_text": "return left - right",
                        "new_text": "return left + right",
                    }
                ],
            }
        ]
    )
    app = create_app(
        HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox)
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks",
            json={
                "repository": str(sample_repository),
                "goal": "Fix calculator addition behavior",
                "file_scope": ["calculator.py"],
                "verification_commands": ["python -m pytest -q"],
            },
        )
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        fetched = client.get(f"/v1/tasks/{task_id}")
        task_diff = client.get(f"/v1/tasks/{task_id}/diff")
        task_trace = client.get(f"/v1/tasks/{task_id}/trace")

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "succeeded"
    assert task_diff.status_code == 200
    assert "+    return left + right" in task_diff.json()["diff"]
    assert task_trace.status_code == 200
    assert [event["node"] for event in task_trace.json()["events"]] == [
        "scout",
        "code",
        "verify",
        "review",
    ]
    assert len(task_trace.json()["checkpoint_revisions"]) >= 5
    assert task_trace.json()["restricted_react"] is None
    assert task_trace.json()["repairs"] == []
    assert task_trace.json()["collaboration"] == []


def test_api_returns_404_for_unknown_task(tmp_path: Path):
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([]), tmp_path / "data", runner_factory=LocalCommandSandbox
        )
    )
    with TestClient(app) as client:
        response = client.get("/v1/tasks/missing")
    assert response.status_code == 404


def test_promotion_conflict_is_audited_and_counted(
    tmp_path: Path, sample_repository: Path
) -> None:
    model = ScriptedModel(
        [
            {
                "summary": "Fix add",
                "edits": [
                    {
                        "path": "calculator.py",
                        "old_text": "return left - right",
                        "new_text": "return left + right",
                    }
                ],
            }
        ]
    )
    governance = GovernanceStore(tmp_path / "data" / "governance.sqlite3")
    app = create_app(
        HarnessOrchestrator(
            model, tmp_path / "data", runner_factory=LocalCommandSandbox
        ),
        governance_store=governance,
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/tasks",
            json={
                "repository": str(sample_repository),
                "goal": "Fix calculator addition behavior",
                "file_scope": ["calculator.py"],
                "verification_commands": ["python -m pytest -q"],
            },
        )
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        (sample_repository / "calculator.py").write_text(
            "def add(left, right):\n    return 100\n", encoding="utf-8"
        )

        conflict = client.post(f"/v1/tasks/{task_id}/promote")
        metrics = client.get("/metrics")
        audit = client.get("/v1/audit/events")

    assert conflict.status_code == 409
    assert 'ai_harness_promotions_total{mode="local",outcome="conflict"} 1' in metrics.text
    assert any(
        event["action"] == "task.promote" and event["outcome"] == "denied"
        for event in audit.json()
    )


def test_readiness_fails_while_worker_circuit_is_open(tmp_path: Path):
    orchestrator = HarnessOrchestrator(
        ScriptedModel([]), tmp_path / "data", runner_factory=LocalCommandSandbox
    )
    store = TaskJobStore(tmp_path / "data" / "jobs.sqlite3")
    pool = TaskWorkerPool(
        orchestrator,
        store,
        max_workers=1,
        poll_interval_seconds=0.01,
        circuit_failure_threshold=1,
        circuit_recovery_seconds=60,
    )
    pool.circuit.record_failure()
    app = create_app(orchestrator, job_store=store, worker_pool=pool)

    with TestClient(app) as client:
        ready = client.get("/ready")
        runtime = client.get("/v1/runtime")

    assert ready.status_code == 503
    assert runtime.json()["ready"] is False
    assert runtime.json()["circuit"]["state"] == "open"


def test_request_id_is_returned_and_untrusted_value_is_replaced(tmp_path: Path):
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([]), tmp_path / "data", runner_factory=LocalCommandSandbox
        ),
        max_job_workers=1,
    )

    with TestClient(app) as client:
        accepted = client.get("/health", headers={"X-Request-ID": "trace-123"})
        replaced = client.get("/health", headers={"X-Request-ID": "unsafe value"})

    assert accepted.headers["X-Request-ID"] == "trace-123"
    assert len(replaced.headers["X-Request-ID"]) == 32
    assert replaced.headers["X-Request-ID"].isalnum()
    assert accepted.headers["X-Content-Type-Options"] == "nosniff"
    assert accepted.headers["X-Frame-Options"] == "DENY"
    assert accepted.headers["Referrer-Policy"] == "no-referrer"
    assert accepted.headers["Cache-Control"] == "no-store"


def test_cors_only_allows_configured_console_origin(tmp_path: Path):
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([]), tmp_path / "data", runner_factory=LocalCommandSandbox
        ),
        max_job_workers=1,
        cors_allowed_origins=["https://console.example.com"],
    )

    with TestClient(app) as client:
        allowed = client.options(
            "/health",
            headers={
                "Origin": "https://console.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        denied = client.options(
            "/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["Access-Control-Allow-Origin"] == "https://console.example.com"
    assert denied.status_code == 400
    assert "Access-Control-Allow-Origin" not in denied.headers


def test_readiness_fails_closed_when_a_storage_dependency_is_unavailable(
    tmp_path: Path,
) -> None:
    orchestrator = HarnessOrchestrator(
        ScriptedModel([]), tmp_path / "data", runner_factory=LocalCommandSandbox
    )
    store = TaskJobStore(tmp_path / "data" / "jobs.sqlite3")
    store.healthcheck = lambda: False  # type: ignore[method-assign]
    app = create_app(orchestrator, job_store=store)

    with TestClient(app) as client:
        ready = client.get("/ready")
        metrics = client.get("/metrics")

    assert ready.status_code == 503
    assert ready.json()["detail"] == "dependencies unavailable: jobs"
    assert 'ai_harness_dependency_ready{dependency="jobs"} 0' in metrics.text


def test_api_requires_and_records_human_approval(tmp_path: Path, sample_repository: Path):
    edit = {
        "summary": "Fix add",
        "edits": [
            {
                "path": "calculator.py",
                "old_text": "return left - right",
                "new_text": "return left + right",
            }
        ],
    }
    model = ScriptedModel([edit, edit])
    app = create_app(
        HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox)
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/tasks",
            json={
                "repository": str(sample_repository),
                "goal": "Modify production calculator behavior safely",
                "file_scope": ["calculator.py"],
                "verification_commands": ["python -m pytest -q"],
            },
        )
        task_id = created.json()["task_id"]
        blocked_resume = client.post(f"/v1/tasks/{task_id}/resume")
        approved = client.post(
            f"/v1/tasks/{task_id}/approval",
            json={
                "approved": True,
                "actor": "web-reviewer",
                "note": "Approved after scope review",
            },
        )
        trace = client.get(f"/v1/tasks/{task_id}/trace")

    assert created.status_code == 200
    assert created.json()["status"] == "needs_input"
    assert created.json()["approval"]["status"] == "pending"
    assert blocked_resume.status_code == 409
    assert approved.status_code == 200
    assert approved.json()["status"] == "succeeded"
    assert approved.json()["approval"]["decided_by"] == "web-reviewer"
    assert trace.json()["approval"]["status"] == "approved"


def test_api_runs_durable_job_and_returns_result(tmp_path: Path, sample_repository: Path):
    model = ScriptedModel(
        [
            {
                "summary": "Fix add",
                "edits": [
                    {
                        "path": "calculator.py",
                        "old_text": "return left - right",
                        "new_text": "return left + right",
                    }
                ],
            }
        ]
    )
    snapshots = FilesystemWorkspaceSnapshotStore(tmp_path / "shared-snapshots")
    orchestrator = HarnessOrchestrator(
        model, tmp_path / "data", runner_factory=LocalCommandSandbox
    )
    app = create_app(
        orchestrator,
        max_job_workers=1,
        job_poll_interval_seconds=0.01,
        workspace_store=snapshots,
    )
    with TestClient(app) as client:
        request_body = {
            "repository": str(sample_repository),
            "goal": "Fix calculator addition behavior",
            "file_scope": ["calculator.py"],
            "verification_commands": ["python -m pytest -q"],
        }
        created = client.post(
            "/v1/jobs",
            json=request_body,
            headers={
                "Idempotency-Key": "api-job-1",
                "X-Request-ID": "job-trace-123",
            },
        )
        assert created.status_code == 202
        assert created.json()["status"] == "pending"
        assert created.json()["request_id"] == "job-trace-123"
        job_id = created.json()["job_id"]
        assert created.json()["workspace_snapshot_id"] == job_id
        assert (snapshots.source_path(job_id) / "calculator.py").is_file()
        replay = client.post(
            "/v1/jobs",
            json=request_body,
            headers={"Idempotency-Key": "api-job-1"},
        )
        conflict = client.post(
            "/v1/jobs",
            json={**request_body, "goal": "A different calculator change"},
            headers={"Idempotency-Key": "api-job-1"},
        )
        deadline = time.monotonic() + 5
        body = created.json()
        while body["status"] in {"pending", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            body = client.get(f"/v1/jobs/{job_id}").json()
        metrics = client.get("/v1/jobs/metrics")
        runtime = client.get("/v1/runtime")
        ready = client.get("/ready")
        prometheus = client.get("/metrics")
        events = client.get(f"/v1/jobs/{job_id}/events")

        task = orchestrator.get(body["task_id"])
        assert task is not None
        shutil.rmtree(Path(task.scratchpad["workspace"]))
        sample_repository.rename(tmp_path / "source-moved-after-run")
        restored_diff = client.get(f"/v1/tasks/{body['task_id']}/diff")

    assert body["status"] == "succeeded"
    assert body["attempts"] == 1
    assert body["result"]["changed_files"] == ["calculator.py"]
    assert replay.status_code == 202
    assert replay.json()["job_id"] == job_id
    assert replay.json()["request_id"] == "job-trace-123"
    assert conflict.status_code == 409
    assert [event["event_type"] for event in events.json()] == [
        "queued",
        "claimed",
        "completed",
    ]
    assert events.json()[0]["details"]["request_id"] == "job-trace-123"
    assert restored_diff.status_code == 200
    assert "+    return left + right" in restored_diff.json()["diff"]
    assert metrics.status_code == 200
    assert metrics.json()["counts"]["succeeded"] == 1
    assert metrics.json()["outstanding_jobs"] == 0
    assert runtime.status_code == 200
    assert runtime.json()["ready"] is True
    assert runtime.json()["circuit"]["state"] == "closed"
    assert len(runtime.json()["workers"]) == 1
    assert ready.status_code == 200
    assert prometheus.status_code == 200
    assert 'ai_harness_jobs{status="succeeded"} 1' in prometheus.text
    assert "ai_harness_circuit_state{state=\"closed\"} 1" in prometheus.text
    assert "ai_harness_quota_reconciliation_runs" in prometheus.text
    assert 'ai_harness_dependency_ready{dependency="jobs"} 1' in prometheus.text
    assert "ai_harness_workspace_snapshots 2" in prometheus.text
    assert "ai_harness_workspace_snapshot_bytes" in prometheus.text
    assert "ai_harness_workspace_snapshot_capacity_bytes 10737418240" in prometheus.text
    assert (
        'ai_harness_http_requests_total{method="GET",route="/ready",status="200"} 1'
        in prometheus.text
    )


def test_api_rejects_job_when_workspace_snapshot_capacity_is_exhausted(
    tmp_path: Path, sample_repository: Path
) -> None:
    snapshots = FilesystemWorkspaceSnapshotStore(
        tmp_path / "shared-snapshots", max_bytes=1
    )
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([]), tmp_path / "data", runner_factory=LocalCommandSandbox
        ),
        workspace_store=snapshots,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={
                "repository": str(sample_repository),
                "goal": "Run a task that cannot fit in snapshot storage",
            },
        )

    assert response.status_code == 413
    assert "exceeds storage capacity" in response.json()["detail"]
    assert snapshots.metrics().snapshot_count == 0


def test_api_updates_waiting_job_after_approval(tmp_path: Path, sample_repository: Path):
    edit = {
        "summary": "Fix add",
        "edits": [
            {
                "path": "calculator.py",
                "old_text": "return left - right",
                "new_text": "return left + right",
            }
        ],
    }
    snapshots = FilesystemWorkspaceSnapshotStore(tmp_path / "shared-snapshots")
    orchestrator = HarnessOrchestrator(
        ScriptedModel([edit, edit]),
        tmp_path / "data",
        runner_factory=LocalCommandSandbox,
    )
    app = create_app(
        orchestrator,
        max_job_workers=1,
        job_poll_interval_seconds=0.01,
        workspace_store=snapshots,
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={
                "repository": str(sample_repository),
                "goal": "Modify production calculator behavior safely",
                "file_scope": ["calculator.py"],
                "verification_commands": ["python -m pytest -q"],
            },
        ).json()
        deadline = time.monotonic() + 5
        job = created
        while job["status"] in {"pending", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            job = client.get(f"/v1/jobs/{created['job_id']}").json()
        assert job["status"] == "needs_input"

        waiting = orchestrator.get(job["task_id"])
        assert waiting is not None
        shutil.rmtree(Path(waiting.scratchpad["workspace"]))

        approved = client.post(
            f"/v1/tasks/{job['task_id']}/approval",
            json={"approved": True, "actor": "reviewer", "note": "scope checked"},
        )
        updated = client.get(f"/v1/jobs/{created['job_id']}")

    assert approved.status_code == 200
    assert updated.json()["status"] == "succeeded"
    assert updated.json()["result"]["approval"]["status"] == "approved"


def test_api_cancels_waiting_job_and_its_checkpoint_state(
    tmp_path: Path, sample_repository: Path
) -> None:
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([]),
            tmp_path / "data",
            runner_factory=LocalCommandSandbox,
        ),
        max_job_workers=1,
        job_poll_interval_seconds=0.01,
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={
                "repository": str(sample_repository),
                "goal": "Modify production calculator behavior safely",
                "file_scope": ["calculator.py"],
                "verification_commands": ["python -m pytest -q"],
            },
        ).json()
        deadline = time.monotonic() + 5
        job = created
        while job["status"] in {"pending", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            job = client.get(f"/v1/jobs/{created['job_id']}").json()
        assert job["status"] == "needs_input"

        cancelled = client.post(f"/v1/jobs/{created['job_id']}/cancel")
        task = client.get(f"/v1/tasks/{job['task_id']}")
        late_approval = client.post(
            f"/v1/tasks/{job['task_id']}/approval",
            json={"approved": True, "actor": "reviewer"},
        )

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert task.json()["status"] == "cancelled"
    assert late_approval.status_code == 409


def test_authenticated_governance_enforces_tenant_rbac_and_separation_of_duties(
    tmp_path: Path, sample_repository: Path
):
    def definition(
        key_id: str,
        secret: str,
        tenant_id: str,
        subject: str,
        *roles: Role,
    ) -> APIKeyDefinition:
        return APIKeyDefinition(
            key_id=key_id,
            secret_sha256=hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            tenant_id=tenant_id,
            subject=subject,
            roles=frozenset(roles),
            repository_roots=[str(sample_repository)],
        )

    authenticator = APIKeyAuthenticator(
        enabled=True,
        config=GovernanceConfig(
            keys=[
                definition(
                    "alice-key",
                    "alice-secret",
                    "tenant-a",
                    "alice",
                    Role.SUBMITTER,
                    Role.REVIEWER,
                ),
                definition(
                    "bob-key",
                    "bob-secret",
                    "tenant-a",
                    "bob",
                    Role.REVIEWER,
                ),
                definition(
                    "carol-key",
                    "carol-secret",
                    "tenant-a",
                    "carol",
                    Role.PUBLISHER,
                    Role.AUDITOR,
                    Role.OPERATOR,
                ),
                definition(
                    "mallory-key",
                    "mallory-secret",
                    "tenant-b",
                    "mallory",
                    Role.SUBMITTER,
                ),
            ],
            tenant_quotas=[
                TenantQuotaPolicy(
                    tenant_id="tenant-a",
                    max_outstanding_jobs=1,
                    max_daily_jobs=10,
                    max_daily_tokens=1000,
                    token_reservation=100,
                ),
                TenantQuotaPolicy(
                    tenant_id="tenant-b",
                    max_outstanding_jobs=1,
                    max_daily_jobs=10,
                    max_daily_tokens=1000,
                    token_reservation=100,
                ),
            ],
        ),
    )
    governance = GovernanceStore(tmp_path / "data" / "governance.sqlite3")
    edit = {
        "summary": "Fix add",
        "edits": [
            {
                "path": "calculator.py",
                "old_text": "return left - right",
                "new_text": "return left + right",
            }
        ],
    }
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([edit, edit]),
            tmp_path / "data",
            runner_factory=LocalCommandSandbox,
        ),
        max_job_workers=1,
        job_poll_interval_seconds=0.01,
        authenticator=authenticator,
        governance_store=governance,
    )
    body = {
        "repository": str(sample_repository),
        "goal": "Modify production calculator behavior safely",
        "file_scope": ["calculator.py"],
        "verification_commands": ["python -m pytest -q"],
    }
    alice = {"Authorization": "Bearer alice-secret"}
    bob = {"Authorization": "Bearer bob-secret"}
    carol = {"Authorization": "Bearer carol-secret"}
    mallory = {"Authorization": "Bearer mallory-secret"}

    with TestClient(app) as client:
        assert client.post("/v1/jobs", json=body).status_code == 401
        created = client.post(
            "/v1/jobs",
            json=body,
            headers={**alice, "Idempotency-Key": "governance-task"},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        replay = client.post(
            "/v1/jobs",
            json=body,
            headers={**alice, "Idempotency-Key": "governance-task"},
        )
        assert replay.status_code == 202
        assert replay.json()["job_id"] == job_id
        deadline = time.monotonic() + 5
        job = created.json()
        while job["status"] in {"pending", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            job = client.get(f"/v1/jobs/{job_id}", headers=alice).json()

        assert job["status"] == "needs_input"
        waiting_quota = client.get("/v1/quota", headers=alice)
        assert waiting_quota.json()["outstanding_jobs"] == 1
        assert waiting_quota.json()["daily_jobs"] == 1
        quota_rejected = client.post(
            "/v1/jobs",
            json={**body, "goal": "Modify another production calculator safely"},
            headers={**alice, "Idempotency-Key": "second-task"},
        )
        assert quota_rejected.status_code == 429
        task_id = job["task_id"]
        assert client.get(f"/v1/jobs/{job_id}", headers=mallory).status_code == 403
        self_approval = client.post(
            f"/v1/tasks/{task_id}/approval",
            json={"approved": True, "actor": "alice", "note": "self approval"},
            headers=alice,
        )
        assert self_approval.status_code == 403
        approved = client.post(
            f"/v1/tasks/{task_id}/approval",
            json={"approved": True, "actor": "bob", "note": "reviewed"},
            headers=bob,
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "succeeded"
        assert client.post(f"/v1/tasks/{task_id}/promote", headers=alice).status_code == 403
        promoted = client.post(f"/v1/tasks/{task_id}/promote", headers=carol)
        audit = client.get("/v1/audit/events", headers=carol)
        integrity = client.get("/v1/audit/integrity", headers=carol)
        settled_quota = client.get("/v1/quota", headers=alice)

    assert promoted.status_code == 200
    assert promoted.json()["promoted_files"] == ["calculator.py"]
    assert audit.status_code == 200
    assert integrity.status_code == 200
    assert integrity.json()["valid"] is True
    assert settled_quota.json()["outstanding_jobs"] == 0
    assert settled_quota.json()["daily_jobs"] == 1
    assert settled_quota.json()["charged_tokens"] == 0
    actions = [event["action"] for event in audit.json()]
    assert "job.create" in actions
    assert "task.approval.decide" in actions
    assert "task.promote" in actions
    assert any(
        event["action"] == "task.approval.decide" and event["outcome"] == "denied"
        for event in audit.json()
    )


def test_authenticated_governance_rejects_repository_outside_key_scope(
    tmp_path: Path, sample_repository: Path
):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    definition = APIKeyDefinition(
        key_id="scoped-key",
        secret_sha256=hashlib.sha256(b"scoped-secret").hexdigest(),
        tenant_id="tenant-a",
        subject="alice",
        roles=frozenset({Role.SUBMITTER}),
        repository_roots=[str(allowed)],
    )
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([]),
            tmp_path / "data",
            runner_factory=LocalCommandSandbox,
        ),
        authenticator=APIKeyAuthenticator(
            enabled=True,
            config=GovernanceConfig(keys=[definition]),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={
                "repository": str(sample_repository),
                "goal": "Fix calculator addition behavior",
                "file_scope": ["calculator.py"],
                "verification_commands": ["python -m pytest -q"],
            },
            headers={"Authorization": "Bearer scoped-secret"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "repository is outside the API key allowlist"
