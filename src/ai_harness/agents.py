from __future__ import annotations

import fnmatch
import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Protocol

import httpx

from .models import (
    CodingDecision,
    CommandResult,
    ModelUsage,
    RepairPlan,
    ReviewDecision,
    ReviewFinding,
    SearchHit,
    TaskRequest,
)


class AgentModel(Protocol):
    def complete_json(self, *, system: str, payload: Dict[str, Any]) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class ModelUsageSnapshot:
    calls: int
    request_chars: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass
class TaskUsageAccumulator:
    calls: int = 0
    request_chars: int = 0
    reported_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def snapshot(self) -> ModelUsage:
        complete = self.calls == self.reported_calls
        prompt_tokens = self.prompt_tokens if complete else None
        completion_tokens = self.completion_tokens if complete else None
        return ModelUsage(
            calls=self.calls,
            request_chars=self.request_chars,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            ),
        )


_TASK_USAGE: ContextVar[Optional[TaskUsageAccumulator]] = ContextVar(
    "ai_harness_task_usage",
    default=None,
)


@contextmanager
def capture_task_model_usage() -> Iterator[TaskUsageAccumulator]:
    accumulator = TaskUsageAccumulator()
    token = _TASK_USAGE.set(accumulator)
    try:
        yield accumulator
    finally:
        _TASK_USAGE.reset(token)


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
        self._calls = 0
        self._request_chars = 0
        self._reported_usage = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._usage_lock = threading.Lock()

    def complete_json(self, *, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        encoded_payload = json.dumps(payload, ensure_ascii=False)
        task_usage = _TASK_USAGE.get()
        if task_usage is not None:
            task_usage.calls += 1
            task_usage.request_chars += len(system) + len(encoded_payload)
        with self._usage_lock:
            self._calls += 1
            self._request_chars += len(system) + len(encoded_payload)
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": encoded_payload},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                if task_usage is not None:
                    task_usage.reported_calls += 1
                    task_usage.prompt_tokens += prompt_tokens
                    task_usage.completion_tokens += completion_tokens
                with self._usage_lock:
                    self._reported_usage += 1
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += completion_tokens
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("model response must be a JSON object")
        return parsed

    def usage_snapshot(self) -> ModelUsageSnapshot:
        with self._usage_lock:
            complete_usage = self._calls == self._reported_usage
            prompt_tokens = self._prompt_tokens if complete_usage else None
            completion_tokens = self._completion_tokens if complete_usage else None
            return ModelUsageSnapshot(
                calls=self._calls,
                request_chars=self._request_chars,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=(
                    prompt_tokens + completion_tokens
                    if prompt_tokens is not None and completion_tokens is not None
                    else None
                ),
            )


class CoderAgent:
    SYSTEM = """You are the Coder in a bounded coding harness.
Return JSON only: {"summary": string, "edits": [{"path": relative path,
"old_text": exact unique text or empty for a new file, "new_text": replacement}]}.
Context snippets may show display-only line-number prefixes such as `12: `. Never include those
prefixes in old_text or new_text. Copy old_text exactly from the code after the prefix, include
enough surrounding code to make it unique, and never use empty old_text for an existing file.
Use only files in scope. Prefer the smallest complete change. Never emit commands."""

    def __init__(self, model: AgentModel) -> None:
        self.model = model

    def propose(
        self,
        request: TaskRequest,
        context: List[SearchHit],
        previous_failure: str = "",
        memory_context: Optional[List[Dict[str, Any]]] = None,
        repair_plan: Optional[RepairPlan] = None,
    ) -> CodingDecision:
        payload = {
            "goal": request.goal,
            "acceptance_criteria": [item.model_dump() for item in request.acceptance_criteria],
            "file_scope": request.file_scope,
            "context": [item.model_dump() for item in context],
            "previous_failure": previous_failure,
            "memory_context": memory_context or [],
            "repair_plan": repair_plan.model_dump(mode="json") if repair_plan else None,
        }
        return CodingDecision.model_validate(
            self.model.complete_json(system=self.SYSTEM, payload=payload)
        )


class ReviewerAgent:
    def review(
        self,
        request: TaskRequest,
        changed_files: List[str],
        verification: List[CommandResult],
    ) -> ReviewDecision:
        findings: List[ReviewFinding] = []
        if not changed_files:
            findings.append(
                ReviewFinding(
                    category="no_changes",
                    message="No files were changed",
                )
            )
        for result in verification:
            if result.timed_out:
                findings.append(
                    ReviewFinding(
                        category="timeout",
                        message=f"Verification timed out: {result.command}",
                        command=result.command,
                        evidence=(result.stderr or result.stdout)[-2000:],
                    )
                )
            elif result.command == "apply_edit_preflight" and result.exit_code != 0:
                findings.append(
                    ReviewFinding(
                        category="edit_application",
                        message="The proposed edit batch could not be applied safely",
                        command=result.command,
                        evidence=(result.stderr or result.stdout)[-2000:],
                    )
                )
            elif result.exit_code != 0:
                findings.append(
                    ReviewFinding(
                        category="verification",
                        message=(
                            f"Verification failed with exit code {result.exit_code}: "
                            f"{result.command}"
                        ),
                        command=result.command,
                        evidence=(result.stderr or result.stdout)[-2000:],
                    )
                )
        if request.file_scope:
            for path in changed_files:
                if not any(
                    path == scope
                    or path.startswith(scope.rstrip("/") + "/")
                    or fnmatch.fnmatch(path, scope)
                    for scope in request.file_scope
                ):
                    findings.append(
                        ReviewFinding(
                            category="scope",
                            message=f"Changed file is outside requested scope: {path}",
                            path=path,
                        )
                    )
        issues = [finding.message for finding in findings]
        return ReviewDecision(
            approved=not findings,
            summary=(
                "Change is bounded and verified" if not findings else "Change requires revision"
            ),
            issues=issues,
            findings=findings,
        )
