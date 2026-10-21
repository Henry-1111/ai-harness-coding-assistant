from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol

import httpx

from .models import CodingDecision, ReviewDecision, SearchHit, TaskRequest


class AgentModel(Protocol):
    def complete_json(self, *, system: str, payload: Dict[str, Any]) -> Dict[str, Any]: ...


class OpenAICompatibleModel:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete_json(self, *, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("model response must be a JSON object")
        return parsed


class CoderAgent:
    SYSTEM = """You are the Coder in a bounded coding harness.
Return JSON only: {"summary": string, "edits": [{"path": relative path,
"old_text": exact unique text or empty for a new file, "new_text": replacement}]}.
Use only files in scope. Prefer the smallest complete change. Never emit commands."""

    def __init__(self, model: AgentModel) -> None:
        self.model = model

    def propose(
        self,
        request: TaskRequest,
        context: List[SearchHit],
        previous_failure: str = "",
        memory_context: Optional[List[Dict[str, Any]]] = None,
    ) -> CodingDecision:
        payload = {
            "goal": request.goal,
            "acceptance_criteria": [item.model_dump() for item in request.acceptance_criteria],
            "file_scope": request.file_scope,
            "context": [item.model_dump() for item in context],
            "previous_failure": previous_failure,
            "memory_context": memory_context or [],
        }
        return CodingDecision.model_validate(
            self.model.complete_json(system=self.SYSTEM, payload=payload)
        )


class ReviewerAgent:
    def review(
        self,
        request: TaskRequest,
        changed_files: List[str],
        verification_exit_codes: List[int],
    ) -> ReviewDecision:
        issues: List[str] = []
        if not changed_files:
            issues.append("No files were changed")
        if any(code != 0 for code in verification_exit_codes):
            issues.append("One or more verification commands failed")
        if request.file_scope:
            for path in changed_files:
                if not any(
                    path == scope or path.startswith(scope.rstrip("/") + "/")
                    for scope in request.file_scope
                ):
                    issues.append(f"Changed file is outside requested scope: {path}")
        return ReviewDecision(
            approved=not issues,
            summary="Change is bounded and verified" if not issues else "Change requires revision",
            issues=issues,
        )
