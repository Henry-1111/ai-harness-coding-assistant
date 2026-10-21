from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest


class ScriptedModel:
    def __init__(self, responses: Iterable[Dict[str, Any]]) -> None:
        self.responses = deque(responses)
        self.calls = []

    def complete_json(self, *, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"system": system, "payload": payload})
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.popleft()


@pytest.fixture
def sample_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "calculator.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    (repository / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return repository
