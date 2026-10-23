import time
from pathlib import Path

from conftest import ScriptedModel
from fastapi.testclient import TestClient

from ai_harness.api import create_app
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
