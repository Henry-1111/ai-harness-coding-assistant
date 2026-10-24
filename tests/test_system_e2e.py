from __future__ import annotations

import json
import threading
import time
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient

from ai_harness.agents import OpenAICompatibleModel
from ai_harness.api import create_app
from ai_harness.orchestrator import HarnessOrchestrator
from ai_harness.sandbox import LocalCommandSandbox


class GatewayStub:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = deque(statuses)
        self.requests: list[dict[str, object]] = []
        self.lock = threading.Lock()


@contextmanager
def gateway_stub(statuses: list[int]) -> Iterator[tuple[str, GatewayStub]]:
    state = GatewayStub(statuses)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            with state.lock:
                state.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "payload": payload,
                    }
                )
                status = state.statuses.popleft() if state.statuses else 200
            if status != 200:
                body = {"error": {"message": "injected upstream failure"}}
            else:
                decision = {
                    "summary": "Fix calculator addition",
                    "edits": [
                        {
                            "path": "calculator.py",
                            "old_text": "return left - right",
                            "new_text": "return left + right",
                        }
                    ],
                }
                body = {
                    "choices": [{"message": {"content": json.dumps(decision)}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                }
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def wait_for_terminal_job(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 10
    body: dict[str, object] = {"status": "pending"}
    while body["status"] in {"pending", "running"} and time.monotonic() < deadline:
        time.sleep(0.02)
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
    return body


def request_body(repository: Path) -> dict[str, object]:
    return {
        "repository": str(repository),
        "goal": "Fix calculator addition behavior",
        "file_scope": ["calculator.py"],
        "verification_commands": ["python -m pytest -q"],
    }


def build_app(base_url: str, data_directory: Path):
    model = OpenAICompatibleModel(
        base_url,
        "gateway-tenant-key",
        "deepseek-chat",
        timeout_seconds=2,
    )
    orchestrator = HarnessOrchestrator(
        model,
        data_directory,
        runner_factory=LocalCommandSandbox,
    )
    return create_app(
        orchestrator,
        max_job_workers=1,
        job_poll_interval_seconds=0.01,
    )


def test_full_http_gateway_job_flow_and_explicit_promotion(
    tmp_path: Path, sample_repository: Path
) -> None:
    with gateway_stub([200]) as (base_url, gateway):
        app = build_app(base_url, tmp_path / "data")
        with TestClient(app) as client:
            created = client.post(
                "/v1/jobs",
                json=request_body(sample_repository),
                headers={"Idempotency-Key": "gateway-e2e-success"},
            )
            assert created.status_code == 202
            job = wait_for_terminal_job(client, created.json()["job_id"])

            assert job["status"] == "succeeded"
            result = job["result"]
            assert isinstance(result, dict)
            assert result["changed_files"] == ["calculator.py"]
            assert result["model_usage"]["total_tokens"] == 30
            assert "left - right" in (sample_repository / "calculator.py").read_text()

            promoted = client.post(f"/v1/tasks/{job['task_id']}/promote")
            assert promoted.status_code == 200
            assert promoted.json()["promoted_files"] == ["calculator.py"]
            metrics = client.get("/metrics")
            assert (
                'ai_harness_promotions_total{mode="local",outcome="success"} 1'
                in metrics.text
            )

    assert "left + right" in (sample_repository / "calculator.py").read_text()
    assert len(gateway.requests) == 1
    request = gateway.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer gateway-tenant-key"
    payload = request["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "deepseek-chat"


def test_http_500_is_retried_and_job_audit_records_recovery(
    tmp_path: Path, sample_repository: Path
) -> None:
    with gateway_stub([500, 200]) as (base_url, gateway):
        app = build_app(base_url, tmp_path / "data")
        with TestClient(app) as client:
            created = client.post(
                "/v1/jobs",
                json=request_body(sample_repository),
                headers={"Idempotency-Key": "gateway-e2e-retry"},
            )
            assert created.status_code == 202
            job_id = created.json()["job_id"]
            job = wait_for_terminal_job(client, job_id)
            events = client.get(f"/v1/jobs/{job_id}/events")

    assert job["status"] == "succeeded"
    assert job["attempts"] == 2
    assert [event["event_type"] for event in events.json()] == [
        "queued",
        "claimed",
        "retry_scheduled",
        "claimed",
        "completed",
    ]
    assert len(gateway.requests) == 2
    assert "left - right" in (sample_repository / "calculator.py").read_text()
