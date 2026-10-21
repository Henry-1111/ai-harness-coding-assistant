from pathlib import Path

from conftest import ScriptedModel
from fastapi.testclient import TestClient

from ai_harness.api import create_app
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

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "succeeded"


def test_api_returns_404_for_unknown_task(tmp_path: Path):
    app = create_app(
        HarnessOrchestrator(
            ScriptedModel([]), tmp_path / "data", runner_factory=LocalCommandSandbox
        )
    )
    with TestClient(app) as client:
        response = client.get("/v1/tasks/missing")
    assert response.status_code == 404
