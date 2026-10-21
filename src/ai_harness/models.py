from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentRole(str, Enum):
    SCOUT = "scout"
    CODER = "coder"
    REVIEWER = "reviewer"


class AcceptanceCriterion(BaseModel):
    description: str = Field(min_length=3, max_length=500)
    command: Optional[str] = Field(default=None, max_length=500)


class TaskRequest(BaseModel):
    repository: str = Field(min_length=1)
    goal: str = Field(min_length=5, max_length=4000)
    acceptance_criteria: List[AcceptanceCriterion] = Field(default_factory=list)
    file_scope: List[str] = Field(default_factory=list)
    verification_commands: List[str] = Field(default_factory=lambda: ["pytest -q"])
    max_replans: int = Field(default=1, ge=0, le=3)

    @field_validator("file_scope")
    @classmethod
    def validate_scope(cls, values: List[str]) -> List[str]:
        for value in values:
            if value.startswith("/") or ".." in value.split("/"):
                raise ValueError("file_scope must contain repository-relative paths")
        return values


class PlanStep(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    role: AgentRole
    depends_on: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    acceptance: List[str] = Field(default_factory=list)


class PlanDAG(BaseModel):
    steps: List[PlanStep]

    @model_validator(mode="after")
    def validate_graph(self) -> "PlanDAG":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        known = set(ids)
        for step in self.steps:
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError(f"step {step.id} has unknown dependencies: {sorted(missing)}")
            if step.id in step.depends_on:
                raise ValueError(f"step {step.id} cannot depend on itself")
        self.topological_order()
        return self

    def topological_order(self) -> List[PlanStep]:
        by_id = {step.id: step for step in self.steps}
        pending = {step.id: set(step.depends_on) for step in self.steps}
        ordered: List[PlanStep] = []
        while pending:
            ready = sorted(step_id for step_id, deps in pending.items() if not deps)
            if not ready:
                raise ValueError("plan contains a dependency cycle")
            for step_id in ready:
                ordered.append(by_id[step_id])
                pending.pop(step_id)
                for dependencies in pending.values():
                    dependencies.discard(step_id)
        return ordered


class RepoSymbol(BaseModel):
    name: str
    kind: Literal["class", "function", "async_function"]
    line: int


class RepoFile(BaseModel):
    path: str
    language: str
    size: int
    symbols: List[RepoSymbol] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    path: str
    line: int
    text: str
    score: float = 0.0
    source: str = "text"


class FileEdit(BaseModel):
    path: str
    old_text: str
    new_text: str

    @field_validator("path")
    @classmethod
    def relative_safe_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("edit path must be repository-relative")
        return value


class CodingDecision(BaseModel):
    summary: str
    edits: List[FileEdit]


class CommandResult(BaseModel):
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False


class ReviewDecision(BaseModel):
    approved: bool
    summary: str
    issues: List[str] = Field(default_factory=list)


class TaskState(BaseModel):
    task_id: str
    request: TaskRequest
    status: TaskStatus = TaskStatus.PENDING
    plan: Optional[PlanDAG] = None
    repo_map: List[RepoFile] = Field(default_factory=list)
    context: List[SearchHit] = Field(default_factory=list)
    changed_files: List[str] = Field(default_factory=list)
    verification: List[CommandResult] = Field(default_factory=list)
    review: Optional[ReviewDecision] = None
    current_node: str = "start"
    replan_count: int = 0
    error: Optional[str] = None
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    changed_files: List[str]
    verification: List[CommandResult]
    review: Optional[ReviewDecision] = None
    error: Optional[str] = None
