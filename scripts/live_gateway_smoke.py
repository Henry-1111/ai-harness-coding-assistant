from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from ai_harness.agents import OpenAICompatibleModel
from ai_harness.config import Settings
from ai_harness.models import TaskRequest, TaskStatus
from ai_harness.orchestrator import HarnessOrchestrator
from ai_harness.sandbox import LocalCommandSandbox


def main() -> int:
    if os.getenv("AI_HARNESS_LIVE_E2E") != "1":
        print("Refusing to spend model quota: set AI_HARNESS_LIVE_E2E=1 explicitly.")
        return 2

    settings = Settings.from_environment()
    if not settings.model_api_key:
        print("AI_HARNESS_MODEL_API_KEY is required.")
        return 2

    with tempfile.TemporaryDirectory(prefix="ai-harness-live-e2e-") as directory:
        root = Path(directory)
        repository = root / "repository"
        repository.mkdir()
        source = "def add(left, right):\n    return left - right\n"
        (repository / "calculator.py").write_text(source, encoding="utf-8")
        (repository / "test_calculator.py").write_text(
            "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        model = OpenAICompatibleModel(
            settings.model_base_url,
            settings.model_api_key,
            settings.model_name,
            settings.model_timeout_seconds,
        )
        orchestrator = HarnessOrchestrator(
            model,
            root / "data",
            runner_factory=LocalCommandSandbox,
        )
        result = orchestrator.run(
            TaskRequest(
                repository=str(repository),
                goal="Fix calculator.py so add(2, 3) returns 5",
                file_scope=["calculator.py"],
                verification_commands=["python -m pytest -q"],
            )
        )
        usage = model.usage_snapshot()
        report = {
            "base_url": settings.model_base_url,
            "model": settings.model_name,
            "status": result.status.value,
            "task_id": result.task_id,
            "changed_files": result.changed_files,
            "model_usage": {
                "calls": usage.calls,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "source_repository_unchanged": (
                repository.joinpath("calculator.py").read_text(encoding="utf-8") == source
            ),
            "error": result.error,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if result.status == TaskStatus.SUCCEEDED else 1


if __name__ == "__main__":
    sys.exit(main())
