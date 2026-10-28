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
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    SCOUT = "scout"
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"


class TaskType(str, Enum):
    BUGFIX = "bugfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    MIGRATION = "migration"
    TEST = "test"
    GENERAL = "general"


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RouteKind(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    COMPLEX = "complex"
    HUMAN_GATE = "human_gate"


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


class NormalizedTask(BaseModel):
    normalized_goal: str
    task_type: TaskType
    complexity: TaskComplexity
    risk_level: RiskLevel
    derived_acceptance: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    requires_human_approval: bool = False


class RouteDecision(BaseModel):
    route: RouteKind
    reasons: List[str] = Field(default_factory=list)
    requires_planning: bool = False
    required_roles: List[AgentRole] = Field(default_factory=list)


class CollaborationPolicy(BaseModel):
    enabled: bool = False
    strategy: Literal["single", "implementer_reviewer"] = "single"
    reasons: List[str] = Field(default_factory=list)
    max_rounds: int = Field(default=0, ge=0, le=3)
    max_parallel_agents: int = Field(default=2, ge=2, le=4)
    model_call_budget: int = Field(default=1, ge=1, le=18)

    @model_validator(mode="after")
    def consistent_budget(self) -> "CollaborationPolicy":
        if self.enabled:
            if self.strategy == "single" or self.max_rounds == 0:
                raise ValueError("enabled collaboration requires a multi-agent strategy")
            if self.model_call_budget < self.max_rounds * 2:
                raise ValueError("model call budget must cover both agents in every round")
        elif self.strategy != "single" or self.max_rounds != 0:
            raise ValueError("disabled collaboration must use the single strategy")
        return self


class AgentAssignment(BaseModel):
    assignment_id: str
    step_id: str
    role: AgentRole
    objective: str = ""
    allowed_paths: List[str] = Field(default_factory=list)
    can_edit: bool = True
    status: Literal["pending", "running", "completed", "failed"] = "pending"


MilestoneCode = Literal["CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CP7", "CP8"]
MilestoneStatus = Literal["completed", "failed"]


class TaskMilestone(BaseModel):
    code: MilestoneCode
    title: str
    status: MilestoneStatus
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PlanStep(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    role: AgentRole
    depends_on: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    acceptance: List[str] = Field(default_factory=list)
    allowed_paths: List[str] = Field(default_factory=list)
    can_edit: bool = True


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
    kind: Literal[
        "class", "function", "async_function", "method", "interface", "type",
        "struct", "enum", "record",
    ]
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


class RetrievalStageTrace(BaseModel):
    stage: Literal[
        "rewrite",
        "forge",
        "focus",
        "semantic",
        "deepen",
        "lsp",
        "judge",
        "refine",
        "select",
    ]
    round: int = Field(default=1, ge=1)
    candidate_count: int = 0
    selected_paths: List[str] = Field(default_factory=list)
    duration_ms: int = 0
    details: Dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    queries: List[str] = Field(default_factory=list)
    stages: List[RetrievalStageTrace] = Field(default_factory=list)
    cache_hit: bool = False
    query_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    context_chars: int = 0
    repository_chars: int = 0
    context_reduction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    stop_reason: str = "candidates_exhausted"
    rounds: int = Field(default=1, ge=1)


class AgentTraceEvent(BaseModel):
    node: str
    outcome: Literal["succeeded", "failed", "replan"]
    duration_ms: int = 0
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class FileEdit(BaseModel):
    path: str
    old_text: str
    new_text: str
    start_line: Optional[int] = Field(default=None, ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)

    @field_validator("path")
    @classmethod
    def relative_safe_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("edit path must be repository-relative")
        return value

    @model_validator(mode="after")
    def complete_line_range(self) -> "FileEdit":
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("start_line and end_line must be provided together")
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class CodingDecision(BaseModel):
    summary: str
    edits: List[FileEdit]


class CollaborationRoleAssignment(BaseModel):
    agent: str = Field(pattern=r"^(implementer(?:_[1-9][0-9]*)?|test_specialist)$")
    role: Optional[Literal["implementer", "test_specialist"]] = None
    objective: str
    allowed_paths: List[str] = Field(default_factory=list)
    can_edit: bool = True
    call_budget: int = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def infer_legacy_role(self) -> "CollaborationRoleAssignment":
        if self.role is None:
            self.role = (
                "test_specialist"
                if self.agent == "test_specialist"
                else "implementer"
            )
        return self


class ProposalEvaluation(BaseModel):
    agent: str
    score: int
    valid: bool
    reasons: List[str] = Field(default_factory=list)


class CollaborationTurn(BaseModel):
    round: int = Field(ge=1)
    implementation_proposal: CodingDecision
    test_proposal: Optional[CodingDecision] = None
    integrated_decision: CodingDecision
    detected_conflicts: List[str] = Field(default_factory=list)
    participants: List[str] = Field(
        default_factory=lambda: ["implementer", "test_specialist"]
    )
    parallel_proposals: bool = False
    arbitrated: bool = False
    model_calls: int = Field(default=2, ge=2, le=9)
    assignments: List[CollaborationRoleAssignment] = Field(default_factory=list)
    role_attempts: Dict[str, int] = Field(default_factory=dict)
    subtask_proposals: Dict[str, CodingDecision] = Field(default_factory=dict)
    original_proposals: Dict[str, CodingDecision] = Field(default_factory=dict)
    competing_proposals: Dict[str, CodingDecision] = Field(default_factory=dict)
    proposal_evaluations: List[ProposalEvaluation] = Field(default_factory=list)
    selected_variants: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ApprovalRecord(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    reason: str = Field(min_length=1, max_length=1000)
    requested_at: datetime = Field(default_factory=utc_now)
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = Field(default=None, min_length=1, max_length=200)
    note: Optional[str] = Field(default=None, max_length=1000)


class CommandResult(BaseModel):
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False


class ReviewFinding(BaseModel):
    category: Literal[
        "no_changes",
        "verification",
        "timeout",
        "scope",
        "edit_application",
        "repeated_failure",
    ]
    severity: Literal["warning", "error"] = "error"
    message: str
    command: Optional[str] = None
    path: Optional[str] = None
    evidence: str = Field(default="", max_length=2000)


class RepairPlan(BaseModel):
    attempt: int = Field(ge=1)
    failure_fingerprint: str
    summary: str
    instructions: List[str] = Field(default_factory=list)
    failed_commands: List[str] = Field(default_factory=list)
    repeat_count: int = Field(default=0, ge=0)
    can_retry: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class ReviewDecision(BaseModel):
    approved: bool
    summary: str
    issues: List[str] = Field(default_factory=list)
    findings: List[ReviewFinding] = Field(default_factory=list)


class ModelUsage(BaseModel):
    calls: int = 0
    request_chars: int = 0
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0


class TaskState(BaseModel):
    task_id: str
    request: TaskRequest
    status: TaskStatus = TaskStatus.PENDING
    normalized: Optional[NormalizedTask] = None
    route: Optional[RouteDecision] = None
    collaboration_policy: CollaborationPolicy = Field(
        default_factory=CollaborationPolicy
    )
    plan: Optional[PlanDAG] = None
    assignments: List[AgentAssignment] = Field(default_factory=list)
    milestones: List[TaskMilestone] = Field(default_factory=list)
    repo_map: List[RepoFile] = Field(default_factory=list)
    context: List[SearchHit] = Field(default_factory=list)
    retrieval_trace: Optional[RetrievalTrace] = None
    trace: List[AgentTraceEvent] = Field(default_factory=list)
    changed_files: List[str] = Field(default_factory=list)
    verification: List[CommandResult] = Field(default_factory=list)
    review: Optional[ReviewDecision] = None
    current_node: str = "start"
    replan_count: int = 0
    repair_history: List[RepairPlan] = Field(default_factory=list)
    collaboration_history: List[CollaborationTurn] = Field(default_factory=list)
    approval: Optional[ApprovalRecord] = None
    model_usage: ModelUsage = Field(default_factory=ModelUsage)
    error: Optional[str] = None
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    normalized: Optional[NormalizedTask] = None
    route: Optional[RouteDecision] = None
    collaboration_policy: CollaborationPolicy = Field(
        default_factory=CollaborationPolicy
    )
    milestones: List[TaskMilestone] = Field(default_factory=list)
    changed_files: List[str]
    verification: List[CommandResult]
    review: Optional[ReviewDecision] = None
    repair_history: List[RepairPlan] = Field(default_factory=list)
    collaboration_history: List[CollaborationTurn] = Field(default_factory=list)
    approval: Optional[ApprovalRecord] = None
    model_usage: ModelUsage = Field(default_factory=ModelUsage)
    retrieval_trace: Optional[RetrievalTrace] = None
    trace: List[AgentTraceEvent] = Field(default_factory=list)
    error: Optional[str] = None
