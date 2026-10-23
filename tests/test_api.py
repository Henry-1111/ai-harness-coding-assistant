import hashlib
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
)
from ai_harness.jobs import TaskJobStore, TaskWorkerPool
from ai_harness.orchestrator import HarnessOrchestrator
from ai_harness.sandbox import LocalCommandSandbox


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
    app = create_app(
        HarnessOrchestrator(model, tmp_path / "data", runner_factory=LocalCommandSandbox),
        max_job_workers=1,
        job_poll_interval_seconds=0.01,
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
            headers={"Idempotency-Key": "api-job-1"},
        )
        assert created.status_code == 202
        assert created.json()["status"] == "pending"
        job_id = created.json()["job_id"]
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

    assert body["status"] == "succeeded"
    assert body["attempts"] == 1
    assert body["result"]["changed_files"] == ["calculator.py"]
    assert replay.status_code == 202
    assert replay.json()["job_id"] == job_id
    assert conflict.status_code == 409
    assert [event["event_type"] for event in events.json()] == [
        "queued",
        "claimed",
        "completed",
    ]
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
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([edit, edit]),
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

        approved = client.post(
            f"/v1/tasks/{job['task_id']}/approval",
            json={"approved": True, "actor": "reviewer", "note": "scope checked"},
        )
        updated = client.get(f"/v1/jobs/{created['job_id']}")

    assert approved.status_code == 200
    assert updated.json()["status"] == "succeeded"
    assert updated.json()["result"]["approval"]["status"] == "approved"


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
            ]
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
        created = client.post("/v1/jobs", json=body, headers=alice)
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        deadline = time.monotonic() + 5
        job = created.json()
        while job["status"] in {"pending", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            job = client.get(f"/v1/jobs/{job_id}", headers=alice).json()

        assert job["status"] == "needs_input"
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

    assert promoted.status_code == 200
    assert promoted.json()["promoted_files"] == ["calculator.py"]
    assert audit.status_code == 200
    assert integrity.status_code == 200
    assert integrity.json()["valid"] is True
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
