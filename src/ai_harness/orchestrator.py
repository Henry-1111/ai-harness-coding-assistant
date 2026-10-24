from __future__ import annotations

import os
import shutil
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from difflib import unified_diff
from fcntl import LOCK_EX, LOCK_UN, flock
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import AgentModel, CoderAgent, ReviewerAgent, capture_task_model_usage
from .checkpoints import CheckpointStore, CheckpointStoreProtocol
from .collaboration import CollaborativeCoder
from .embeddings import EmbeddingProvider
from .git_tools import GitCommandError, GitRepository
from .knowledge import (
    KnowledgeIdentity,
    RestrictedKnowledgeAgent,
    repository_knowledge_registry,
    sources_to_hits,
)
from .mcp import MCPToolCatalog
from .memory import LayeredMemoryStore, LayeredMemoryStoreProtocol, MemoryLayer
from .models import (
    AgentTraceEvent,
    ApprovalRecord,
    MilestoneCode,
    MilestoneStatus,
    ReviewFinding,
    RouteKind,
    TaskMilestone,
    TaskRequest,
    TaskResult,
    TaskState,
    TaskStatus,
    utc_now,
)
from .planner import TaskRouter
from .promotion import PromotionPublisher
from .repository import (
    DefinitionProvider,
    ProgressiveRetriever,
    RepoMapBuilder,
    Repository,
    RetrievalCache,
)
from .sandbox import CommandRunner, DockerSandbox
from .self_healing import SelfHealingPlanner
from .tools import ToolContext, default_registry
from .workspaces import IGNORED_WORKSPACE_NAMES, WorkspaceSnapshotStore


class WorkflowState(TypedDict):
    task: TaskState


@dataclass(frozen=True)
class ExecutionControl:
    should_cancel: Callable[[], bool]
    deadline: Optional[float] = None
    tenant_id: Optional[str] = None
    actor: Optional[str] = None


class ExecutionInterrupted(RuntimeError):
    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


_EXECUTION_CONTROL: ContextVar[Optional[ExecutionControl]] = ContextVar(
    "ai_harness_execution_control", default=None
)


class HarnessOrchestrator:
    """LangGraph-backed Scout -> Coder -> Verify -> Reviewer workflow."""

    def __init__(
        self,
        model: AgentModel,
        data_directory: Path,
        *,
        checkpoint_store: Optional[CheckpointStoreProtocol] = None,
        memory_store: Optional[LayeredMemoryStoreProtocol] = None,
        runner_factory: Optional[Callable[[Path], CommandRunner]] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        definition_provider_factory: Optional[Callable[[Path], DefinitionProvider]] = None,
        restricted_react_enabled: bool = False,
        restricted_react_max_tool_calls: int = 3,
        mcp_tool_catalog: Optional[MCPToolCatalog] = None,
        tenant_id: str = "local",
        collaboration_mode: Literal["auto", "disabled", "forced"] = "auto",
        workspace_store: Optional[WorkspaceSnapshotStore] = None,
        promotion_publisher: Optional[PromotionPublisher] = None,
    ) -> None:
        self.model = model
        self.data_directory = data_directory.resolve()
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints = checkpoint_store or CheckpointStore(
            self.data_directory / "checkpoints.sqlite3"
        )
        self.router = TaskRouter()
        self.repo_map_builder = RepoMapBuilder()
        self.retrieval_cache = RetrievalCache()
        self.embedding_provider = embedding_provider
        self.definition_provider_factory = definition_provider_factory
        if not 1 <= restricted_react_max_tool_calls <= 3:
            raise ValueError("restricted ReAct max tool calls must be within 1..3")
        self.restricted_react_enabled = restricted_react_enabled
        self.restricted_react_max_tool_calls = restricted_react_max_tool_calls
        self.mcp_tool_catalog = mcp_tool_catalog
        self.tenant_id = tenant_id
        if collaboration_mode not in {"auto", "disabled", "forced"}:
            raise ValueError("collaboration_mode must be auto, disabled or forced")
        self.collaboration_mode = collaboration_mode
        self.workspace_store = workspace_store
        self.promotion_publisher = promotion_publisher
        self.memory = memory_store or LayeredMemoryStore(self.data_directory / "memory.sqlite3")
        if self.memory.get(MemoryLayer.POLICY, "default", "instructions") is None:
            self.memory.put(
                MemoryLayer.POLICY,
                "default",
                "instructions",
                {"require_tests": True, "prefer_smallest_complete_change": True},
            )
        self.coder = CoderAgent(model)
        self.collaborative_coder = CollaborativeCoder(model)
        self.reviewer = ReviewerAgent()
        self.self_healing = SelfHealingPlanner()
        self.tools = default_registry()
        self.runner_factory = runner_factory or (lambda workspace: DockerSandbox(workspace))

    def run(
        self,
        request: TaskRequest,
        *,
        control: Optional[ExecutionControl] = None,
        workspace_source: Optional[Path] = None,
        workspace_snapshot_id: Optional[str] = None,
    ) -> TaskResult:
        task_id = str(uuid.uuid4())
        workspace = self._create_workspace(
            task_id, workspace_source or Path(request.repository)
        )
        normalized = self.router.normalizer.normalize(request)
        route = self.router.decide(normalized)
        plan = self.router.build_plan(
            request,
            normalized,
            route,
            collaboration=self._should_collaborate(route.route),
        )
        state = TaskState(
            task_id=task_id,
            request=request,
            status=TaskStatus.RUNNING,
            normalized=normalized,
            route=route,
            plan=plan,
            assignments=self.router.assignments(plan),
            scratchpad={
                "workspace": str(workspace),
                "tenant_id": (control.tenant_id if control else None) or self.tenant_id,
                "requested_by": (control.actor if control else None) or "system",
                "source_snapshot_id": workspace_snapshot_id,
                "task_workspace_snapshot_id": f"task-{task_id}",
            },
        )
        self._record_milestone(
            state,
            "CP1",
            "Goal and acceptance conditions normalized",
            "completed",
            {
                "task_type": normalized.task_type.value,
                "complexity": normalized.complexity.value,
                "risk_level": normalized.risk_level.value,
                "requires_clarification": normalized.requires_clarification,
            },
        )
        if normalized.requires_human_approval:
            state.status = TaskStatus.NEEDS_INPUT
            state.current_node = "approval"
            state.approval = ApprovalRecord(
                reason=(
                    f"risk={normalized.risk_level.value}; route={route.route.value}; "
                    "explicit approval is required before model or tool execution"
                )
            )
        self.checkpoints.save(state)
        if state.status == TaskStatus.NEEDS_INPUT:
            self.publish_workspace(state.task_id)
            return self._result(state)
        return self._invoke(state, control=control)

    def resume(self, task_id: str) -> TaskResult:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return self._result(state)
        if state.approval and state.approval.status == "pending":
            raise RuntimeError("task requires an explicit approval decision before resume")
        self._ensure_workspace(state)
        state.status = TaskStatus.RUNNING
        state.error = None
        result = self._invoke(state)
        self.publish_workspace(task_id)
        return result

    def decide_approval(
        self,
        task_id: str,
        *,
        approved: bool,
        actor: str,
        note: str = "",
    ) -> TaskResult:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status == TaskStatus.CANCELLED:
            raise RuntimeError("cancelled tasks cannot be approved or rejected")
        if state.approval is None:
            raise RuntimeError("task does not require approval")
        decision = "approved" if approved else "rejected"
        if state.approval.status != "pending":
            if state.approval.status == decision:
                return self._result(state)
            raise RuntimeError(
                f"approval was already {state.approval.status} by "
                f"{state.approval.decided_by or 'unknown'}"
            )
        actor = actor.strip()
        if not actor:
            raise ValueError("approval actor is required")
        note = note.strip()
        state.approval = ApprovalRecord.model_validate(
            {
                **state.approval.model_dump(),
                "status": decision,
                "decided_at": utc_now(),
                "decided_by": actor,
                "note": note or None,
            }
        )
        if approved:
            state.status = TaskStatus.RUNNING
            state.current_node = "start"
            state.error = None
            state.trace.append(
                AgentTraceEvent(
                    node="approval",
                    outcome="succeeded",
                    details={"decision": decision, "actor": actor, "note": note},
                )
            )
            self.checkpoints.save(state)
            return self._invoke(state)

        state.status = TaskStatus.FAILED
        state.current_node = "failed"
        state.error = f"Approval rejected by {actor}" + (f": {note}" if note else "")
        state.trace.append(
            AgentTraceEvent(
                node="approval",
                outcome="failed",
                details={"decision": decision, "actor": actor, "note": note},
            )
        )
        self._record_milestone(
            state,
            "CP8",
            "Final result accepted",
            "failed",
            {"reason": "approval_rejected", "actor": actor},
        )
        self._record_experience(state, "failed", failure_type="approval_rejected")
        self.checkpoints.save(state)
        return self._result(state)

    def cancel_waiting(self, task_id: str, *, actor: str) -> TaskResult:
        """Cancel a task paused at its approval gate without executing more work."""
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status == TaskStatus.CANCELLED:
            return self._result(state)
        if state.status != TaskStatus.NEEDS_INPUT:
            raise RuntimeError("only tasks waiting for approval can be cancelled directly")
        actor = actor.strip()
        if not actor:
            raise ValueError("cancellation actor is required")
        state.status = TaskStatus.CANCELLED
        state.current_node = "cancelled"
        state.error = f"Task cancelled by {actor} while waiting for approval"
        state.trace.append(
            AgentTraceEvent(
                node="approval",
                outcome="failed",
                details={"decision": "cancelled", "actor": actor},
            )
        )
        self._record_milestone(
            state,
            "CP8",
            "Final result accepted",
            "failed",
            {"reason": "approval_cancelled", "actor": actor},
        )
        self.checkpoints.save(state)
        return self._result(state)

    def get(self, task_id: str) -> Optional[TaskState]:
        return self.checkpoints.load(task_id)

    def promote(self, task_id: str) -> list[str]:
        """Atomically publish reviewed files unless their source baseline changed."""
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status != TaskStatus.SUCCEEDED or not state.review or not state.review.approved:
            raise RuntimeError("only approved tasks can be promoted")
        workspace = self._workspace_repository(state)
        if self.promotion_publisher is not None:
            receipt = self.promotion_publisher.publish(
                task_id,
                workspace.root,
                state.changed_files,
            )
            state.scratchpad["promotion"] = {
                "mode": receipt.mode,
                "branch": receipt.branch,
                "commit": receipt.commit,
                "remote": receipt.remote,
                "created": receipt.created,
            }
            state.scratchpad["promoted_files"] = list(state.changed_files)
            state.scratchpad["promoted_at"] = utc_now().isoformat()
            self.checkpoints.save(state)
            return list(state.changed_files)
        source = Repository(Path(state.request.repository))
        baselines = state.scratchpad.get("promotion_baselines")
        if not isinstance(baselines, dict):
            raise RuntimeError("promotion baseline is unavailable; rerun the task")
        promoted: list[str] = []
        with self._promotion_lock(source.root):
            pending: list[tuple[str, Path, Path]] = []
            conflicts: list[str] = []
            for relative_path in state.changed_files:
                if relative_path not in baselines:
                    raise RuntimeError(
                        f"promotion baseline is unavailable for {relative_path}; rerun the task"
                    )
                workspace_path = workspace.resolve(relative_path)
                destination = source.resolve(relative_path, must_exist=False)
                desired_digest = self._file_digest(workspace_path)
                current_digest = self._file_digest(destination)
                baseline_digest = baselines[relative_path]
                if current_digest == desired_digest:
                    promoted.append(relative_path)
                elif current_digest != baseline_digest:
                    conflicts.append(relative_path)
                else:
                    pending.append((relative_path, workspace_path, destination))
            if conflicts:
                raise RuntimeError(
                    "source files changed after task creation: " + ", ".join(conflicts)
                )
            for relative_path, workspace_path, destination in pending:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.parent / (
                    f".{destination.name}.ai-harness-{uuid.uuid4().hex}.tmp"
                )
                try:
                    shutil.copy2(workspace_path, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                promoted.append(relative_path)
        state.scratchpad["promoted_files"] = list(promoted)
        state.scratchpad["promoted_at"] = utc_now().isoformat()
        self.checkpoints.save(state)
        return promoted

    @contextmanager
    def _promotion_lock(self, repository: Path) -> Iterator[None]:
        git_directory = repository / ".git"
        if git_directory.is_dir():
            lock_path = git_directory / "ai-harness-promote.lock"
        else:
            lock_directory = self.data_directory / "promotion-locks"
            lock_directory.mkdir(parents=True, exist_ok=True)
            lock_name = sha256(str(repository).encode()).hexdigest()
            lock_path = lock_directory / f"{lock_name}.lock"
        with lock_path.open("a+b") as lock_file:
            flock(lock_file.fileno(), LOCK_EX)
            try:
                yield
            finally:
                flock(lock_file.fileno(), LOCK_UN)

    @staticmethod
    def _file_digest(path: Path) -> Optional[str]:
        if not path.is_file():
            return None
        digest = sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def diff(self, task_id: str) -> str:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        source = self._source_repository(state)
        workspace = self._workspace_repository(state)
        chunks = []
        for relative_path in state.changed_files:
            try:
                before = source.read_text(relative_path).splitlines(keepends=True)
            except FileNotFoundError:
                before = []
            after = workspace.read_text(relative_path).splitlines(keepends=True)
            chunks.extend(
                unified_diff(
                    before,
                    after,
                    fromfile=f"a/{relative_path}",
                    tofile=f"b/{relative_path}",
                )
            )
        return "".join(chunks)

    def _invoke(
        self,
        state: TaskState,
        *,
        control: Optional[ExecutionControl] = None,
    ) -> TaskResult:
        graph = self._build_graph()
        token = _EXECUTION_CONTROL.set(control)
        usage = None
        try:
            with capture_task_model_usage() as usage:
                output = graph.invoke({"task": state})
            final_state = output["task"]
        except ExecutionInterrupted as exc:
            failed_node = state.current_node
            state.status = TaskStatus.FAILED if exc.timed_out else TaskStatus.CANCELLED
            state.error = str(exc)
            state.current_node = "failed"
            state.trace.append(
                AgentTraceEvent(
                    node="workflow",
                    outcome="failed",
                    details={
                        "error_type": "task_timeout" if exc.timed_out else "task_cancelled",
                        "message": str(exc),
                    },
                )
            )
            self._set_assignment_status(state, failed_node, "failed")
            self._record_milestone(
                state,
                "CP8",
                "Final result accepted",
                "failed",
                {"reason": "task_timeout" if exc.timed_out else "task_cancelled"},
            )
            self._record_experience(
                state,
                state.status.value,
                failure_type="task_timeout" if exc.timed_out else "task_cancelled",
            )
            self.checkpoints.save(state)
            final_state = state
        except Exception as exc:
            failed_node = state.current_node
            state.status = TaskStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            state.current_node = "failed"
            state.trace.append(
                AgentTraceEvent(
                    node="workflow",
                    outcome="failed",
                    details={"error_type": type(exc).__name__, "message": str(exc)},
                )
            )
            self._set_assignment_status(state, failed_node, "failed")
            self._record_milestone(
                state,
                "CP8",
                "Final result accepted",
                "failed",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
            self._record_experience(state, "failed", failure_type=type(exc).__name__)
            self.checkpoints.save(state)
            final_state = state
        finally:
            _EXECUTION_CONTROL.reset(token)
        if usage is not None:
            current = final_state.model_usage
            captured = usage.snapshot()
            prompt_tokens = (
                current.prompt_tokens + captured.prompt_tokens
                if current.prompt_tokens is not None and captured.prompt_tokens is not None
                else None
            )
            completion_tokens = (
                current.completion_tokens + captured.completion_tokens
                if current.completion_tokens is not None and captured.completion_tokens is not None
                else None
            )
            final_state.model_usage = current.model_copy(
                update={
                    "calls": current.calls + captured.calls,
                    "request_chars": current.request_chars + captured.request_chars,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": (
                        prompt_tokens + completion_tokens
                        if prompt_tokens is not None and completion_tokens is not None
                        else None
                    ),
                }
            )
            self.checkpoints.save(final_state)
        return self._result(final_state)

    @staticmethod
    def _check_execution_control() -> None:
        control = _EXECUTION_CONTROL.get()
        if control is None:
            return
        if control.should_cancel():
            raise ExecutionInterrupted("task cancellation was requested")
        if control.deadline is not None and time.monotonic() >= control.deadline:
            raise ExecutionInterrupted("task execution deadline exceeded", timed_out=True)

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("scout", self._scout)
        graph.add_node("plan", self._plan)
        graph.add_node("code", self._code)
        graph.add_node("verify", self._verify)
        graph.add_node("review", self._review)
        graph.add_conditional_edges(
            START,
            self._resume_route,
            {
                "scout": "scout",
                "plan": "plan",
                "code": "code",
                "verify": "verify",
                "review": "review",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "scout",
            self._after_scout_route,
            {"plan": "plan", "code": "code"},
        )
        graph.add_edge("plan", "code")
        graph.add_edge("code", "verify")
        graph.add_edge("verify", "review")
        graph.add_conditional_edges(
            "review",
            self._review_route,
            {"code": "code", "end": END},
        )
        return graph.compile()

    def _resume_route(
        self,
        payload: WorkflowState,
    ) -> Literal["scout", "plan", "code", "verify", "review", "end"]:
        task = payload["task"]
        current = task.current_node
        if current == "review" and task.status == TaskStatus.RUNNING:
            return "code"
        return {
            "start": "scout",
            "scout": "plan" if task.route and task.route.requires_planning else "code",
            "plan": "code",
            "code": "verify",
            "verify": "review",
        }.get(current, "end")  # type: ignore[return-value]

    @staticmethod
    def _after_scout_route(payload: WorkflowState) -> Literal["plan", "code"]:
        route = payload["task"].route
        return "plan" if route and route.requires_planning else "code"

    @staticmethod
    def _review_route(payload: WorkflowState) -> Literal["code", "end"]:
        task = payload["task"]
        if task.review and task.review.approved:
            return "end"
        if task.status == TaskStatus.RUNNING:
            return "code"
        return "end"

    def _scout(self, state: WorkflowState) -> WorkflowState:
        self._check_execution_control()
        started = time.monotonic()
        task = state["task"]
        execution_tenant = str(task.scratchpad.get("tenant_id", self.tenant_id))
        self._set_assignment_status(task, "scout", "running")
        repository = self._workspace_repository(task)
        task.repo_map = self.repo_map_builder.build(repository, task.request.file_scope)
        changed_paths: list[str] = []
        try:
            changed_paths = [
                change.path for change in GitRepository(Path(task.request.repository)).changes()
            ]
        except (GitCommandError, FileNotFoundError):
            pass
        definition_provider = (
            self.definition_provider_factory(repository.root)
            if self.definition_provider_factory
            else None
        )
        index = ProgressiveRetriever(
            repository,
            task.repo_map,
            changed_paths=changed_paths,
            cache=self.retrieval_cache,
            embedding_provider=self.embedding_provider,
            definition_provider=definition_provider,
        )
        task.context = index.search(task.request.goal, limit=30)
        self._check_execution_control()
        task.retrieval_trace = index.last_trace
        knowledge_result = None
        if self.restricted_react_enabled and index.last_trace.stop_reason != "evidence_sufficient":
            registry = repository_knowledge_registry(repository, task.repo_map)
            project_id = self._project_id(task.request.repository, execution_tenant)
            allowed_tools = {"search_project_knowledge", "get_source_detail"}
            allowed_mcp_servers: frozenset[str] = frozenset()
            if self.mcp_tool_catalog:
                mcp_tools, allowed_mcp_servers = self.mcp_tool_catalog.register_authorized_tools(
                    registry,
                    tenant_id=execution_tenant,
                    project_id=project_id,
                )
                allowed_tools.update(mcp_tools)
            identity = KnowledgeIdentity(
                tenant_id=execution_tenant,
                project_id=project_id,
                allowed_tools=frozenset(allowed_tools),
                allowed_paths=task.request.file_scope,
                allowed_mcp_servers=allowed_mcp_servers,
            )
            knowledge_result = RestrictedKnowledgeAgent(
                self.model,
                registry,
                max_tool_calls=self.restricted_react_max_tool_calls,
            ).run(task.request.goal, identity)
            self._check_execution_control()
            known = {(hit.path, hit.line, hit.text) for hit in task.context}
            for hit in sources_to_hits(knowledge_result.sources):
                key = (hit.path, hit.line, hit.text)
                if key not in known:
                    task.context.append(hit)
                    known.add(key)
            task.scratchpad["restricted_react"] = knowledge_result.model_dump(mode="json")
        task.scratchpad["git_changed_paths"] = changed_paths
        project_id = self._project_id(task.request.repository, execution_tenant)
        self.memory.put(
            MemoryLayer.PROJECT,
            project_id,
            "last_retrieval",
            index.last_trace.model_dump(mode="json"),
        )
        self.memory.put(
            MemoryLayer.SESSION,
            task.task_id,
            "progress",
            {"node": "scout", "context_hits": len(task.context)},
        )
        task.trace.append(
            AgentTraceEvent(
                node="scout",
                outcome="succeeded",
                duration_ms=round((time.monotonic() - started) * 1000),
                details={
                    "repo_files": len(task.repo_map),
                    "context_hits": len(task.context),
                    "stop_reason": index.last_trace.stop_reason,
                    "restricted_react": (
                        {
                            "tool_calls": knowledge_result.tool_calls,
                            "stop_reason": knowledge_result.stop_reason,
                        }
                        if knowledge_result
                        else None
                    ),
                },
            )
        )
        self._set_assignment_status(task, "scout", "completed")
        self._record_milestone(
            task,
            "CP2",
            "Code context and dependencies confirmed",
            "completed",
            {
                "repo_files": len(task.repo_map),
                "context_hits": len(task.context),
                "git_changed_paths": changed_paths,
                "restricted_react_tool_calls": (
                    knowledge_result.tool_calls if knowledge_result else 0
                ),
            },
        )
        if not task.route or not task.route.requires_planning:
            self._record_plan_milestones(task)
        task.current_node = "scout"
        self.checkpoints.save(task)
        return {"task": task}

    def _plan(self, state: WorkflowState) -> WorkflowState:
        self._check_execution_control()
        started = time.monotonic()
        task = state["task"]
        self._set_assignment_status(task, "plan", "running")
        if task.plan is None:
            raise RuntimeError("normalized task does not have an execution plan")
        ordered_steps = task.plan.topological_order()
        assigned_steps = {assignment.step_id for assignment in task.assignments}
        missing_assignments = [step.id for step in ordered_steps if step.id not in assigned_steps]
        if missing_assignments:
            raise RuntimeError(f"plan steps are missing assignments: {missing_assignments}")
        self._record_plan_milestones(task)
        self._set_assignment_status(task, "plan", "completed")
        task.trace.append(
            AgentTraceEvent(
                node="plan",
                outcome="succeeded",
                duration_ms=round((time.monotonic() - started) * 1000),
                details={
                    "route": task.route.route.value if task.route else "unknown",
                    "steps": [step.id for step in ordered_steps],
                    "assignments": len(task.assignments),
                },
            )
        )
        task.current_node = "plan"
        self.checkpoints.save(task)
        return {"task": task}

    def _code(self, state: WorkflowState) -> WorkflowState:
        self._check_execution_control()
        started = time.monotonic()
        task = state["task"]
        task.current_node = "code"
        self._set_assignment_status(task, "code", "running")
        self._record_milestone(
            task,
            "CP5",
            "Code patch write boundary reached",
            "completed",
            {"attempt": task.replan_count + 1, "allowed_paths": task.request.file_scope},
        )
        previous_failure = "\n".join(
            result.stderr or result.stdout for result in task.verification if result.exit_code != 0
        )[-4000:]
        memory_tenant = str(task.scratchpad.get("tenant_id", self.tenant_id))
        project_id = self._project_id(task.request.repository, memory_tenant)
        memory_context = self.memory.resolve(
            "instructions",
            policy_id="default",
            project_id=project_id,
            user_id=str(task.scratchpad.get("requested_by", "system")),
            session_id=task.task_id,
        )
        experiences = self.memory.get(MemoryLayer.PROJECT, project_id, "experiences") or []
        if experiences:
            memory_context.append(
                {
                    "layer": "project_experience",
                    "owner_id": project_id,
                    "value": experiences[-5:],
                }
            )
        repair_plan = task.repair_history[-1] if task.repair_history else None
        collaboration_turn = None
        if task.route and self._should_collaborate(task.route.route):
            self._set_assignment_status(task, "implementation", "running")
            self._set_assignment_status(task, "test_design", "running")
            try:
                collaboration_turn = self.collaborative_coder.propose(
                    task.request,
                    task.context,
                    round_number=len(task.collaboration_history) + 1,
                    previous_failure=previous_failure,
                    memory_context=memory_context,
                    repair_plan=repair_plan,
                )
            except Exception:
                self._set_assignment_status(task, "implementation", "failed")
                self._set_assignment_status(task, "test_design", "failed")
                raise
            self._set_assignment_status(task, "implementation", "completed")
            self._set_assignment_status(task, "test_design", "completed")
            task.collaboration_history.append(collaboration_turn)
            decision = collaboration_turn.integrated_decision
        else:
            decision = self.coder.propose(
                task.request,
                task.context,
                previous_failure,
                memory_context=memory_context,
                repair_plan=repair_plan,
            )
        self._check_execution_control()
        repository = self._workspace_repository(task)
        context = ToolContext(
            repository=repository,
            permissions={"read", "write"},
            allowed_paths=task.request.file_scope,
        )
        for edit in decision.edits:
            self._check_execution_control()
            baselines = task.scratchpad.setdefault("promotion_baselines", {})
            if not isinstance(baselines, dict):
                raise RuntimeError("invalid promotion baseline state")
            if edit.path not in baselines:
                candidate = repository.resolve(edit.path, must_exist=False)
                baselines[edit.path] = self._file_digest(candidate)
            path = self.tools.execute("apply_edit", context, edit=edit)
            if path not in task.changed_files:
                task.changed_files.append(path)
        task.scratchpad["coder_summary"] = decision.summary
        task.verification = []
        task.review = None
        task.trace.append(
            AgentTraceEvent(
                node="code",
                outcome="succeeded",
                duration_ms=round((time.monotonic() - started) * 1000),
                details={
                    "changed_files": list(task.changed_files),
                    "collaboration": (
                        {
                            "round": collaboration_turn.round,
                            "participants": collaboration_turn.participants,
                            "candidate_edits": len(
                                collaboration_turn.implementation_proposal.edits
                            ),
                            "integrated_edits": len(collaboration_turn.integrated_decision.edits),
                            "detected_conflicts": collaboration_turn.detected_conflicts,
                        }
                        if collaboration_turn
                        else None
                    ),
                },
            )
        )
        self._set_assignment_status(task, "code", "completed")
        self.memory.put(
            MemoryLayer.SCRATCHPAD,
            task.task_id,
            "progress",
            {"node": "code", "changed_files": list(task.changed_files)},
        )
        task.current_node = "code"
        self.checkpoints.save(task)
        return {"task": task}

    def _verify(self, state: WorkflowState) -> WorkflowState:
        self._check_execution_control()
        started = time.monotonic()
        task = state["task"]
        self._set_assignment_status(task, "verify", "running")
        workspace = Path(str(task.scratchpad["workspace"]))
        sandbox = self.runner_factory(workspace)
        task.verification = []
        for command in task.request.verification_commands:
            self._check_execution_control()
            task.verification.append(sandbox.run(command))
            self._check_execution_control()
        exit_codes = [result.exit_code for result in task.verification]
        task.trace.append(
            AgentTraceEvent(
                node="verify",
                outcome="succeeded" if all(code == 0 for code in exit_codes) else "failed",
                duration_ms=round((time.monotonic() - started) * 1000),
                details={"commands": len(task.verification), "exit_codes": exit_codes},
            )
        )
        verification_passed = all(code == 0 for code in exit_codes)
        self._set_assignment_status(
            task,
            "verify",
            "completed" if verification_passed else "failed",
        )
        self._record_milestone(
            task,
            "CP6",
            "Patch applied and deterministic tests completed",
            "completed" if verification_passed else "failed",
            {"commands": len(task.verification), "exit_codes": exit_codes},
        )
        self.memory.put(
            MemoryLayer.SESSION,
            task.task_id,
            "progress",
            {"node": "verify", "exit_codes": exit_codes},
        )
        task.current_node = "verify"
        self.checkpoints.save(task)
        return {"task": task}

    def _review(self, state: WorkflowState) -> WorkflowState:
        self._check_execution_control()
        started = time.monotonic()
        task = state["task"]
        self._set_assignment_status(task, "review", "running")
        task.review = self.reviewer.review(
            task.request,
            task.changed_files,
            task.verification,
        )
        repair_plan = None
        if task.review.approved:
            task.status = TaskStatus.SUCCEEDED
        else:
            repair_plan = self.self_healing.build(
                task.review,
                task.verification,
                task.repair_history,
            )
            task.repair_history.append(repair_plan)
        if task.review.approved:
            pass
        elif repair_plan and repair_plan.can_retry and task.replan_count < task.request.max_replans:
            task.replan_count += 1
            task.status = TaskStatus.RUNNING
        else:
            task.status = TaskStatus.FAILED
            if repair_plan and not repair_plan.can_retry:
                repeated = ReviewFinding(
                    category="repeated_failure",
                    message="The same deterministic failure repeated; self-healing stopped early",
                    evidence=repair_plan.failure_fingerprint,
                )
                task.review.findings.append(repeated)
                task.review.issues.append(repeated.message)
            task.error = "; ".join(task.review.issues)
        outcome = (
            "succeeded"
            if task.review.approved
            else "replan"
            if task.status == TaskStatus.RUNNING
            else "failed"
        )
        task.trace.append(
            AgentTraceEvent(
                node="review",
                outcome=outcome,
                duration_ms=round((time.monotonic() - started) * 1000),
                details={
                    "approved": task.review.approved,
                    "issues": list(task.review.issues),
                    "replan_count": task.replan_count,
                    "repair_plan": repair_plan.model_dump(mode="json") if repair_plan else None,
                },
            )
        )
        self._set_assignment_status(
            task,
            "review",
            "completed" if task.review.approved else "failed",
        )
        self._record_milestone(
            task,
            "CP7",
            "Integration verification and reviewer completed",
            "completed" if task.review.approved else "failed",
            {
                "approved": task.review.approved,
                "issues": list(task.review.issues),
                "replan_count": task.replan_count,
                "repair_attempts": len(task.repair_history),
            },
        )
        if task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
            self._record_milestone(
                task,
                "CP8",
                "Final result accepted",
                "completed" if task.status == TaskStatus.SUCCEEDED else "failed",
                {"status": task.status.value},
            )
            self._record_experience(task, task.status.value)
        self.memory.put(
            MemoryLayer.SESSION,
            task.task_id,
            "progress",
            {"node": "review", "outcome": outcome, "replan_count": task.replan_count},
        )
        task.current_node = "review"
        self.checkpoints.save(task)
        return {"task": task}

    def _record_plan_milestones(self, task: TaskState) -> None:
        if task.plan is None:
            raise RuntimeError("task plan is missing")
        ordered_steps = task.plan.topological_order()
        self._record_milestone(
            task,
            "CP3",
            "Plan DAG completed and structurally validated",
            "completed",
            {"steps": [step.id for step in ordered_steps]},
        )
        self._record_milestone(
            task,
            "CP4",
            "Subtask assignments created",
            "completed",
            {
                "assignments": [
                    {
                        "assignment_id": assignment.assignment_id,
                        "step_id": assignment.step_id,
                        "role": assignment.role.value,
                    }
                    for assignment in task.assignments
                ]
            },
        )

    @staticmethod
    def _set_assignment_status(task: TaskState, step_id: str, status: str) -> None:
        for index, assignment in enumerate(task.assignments):
            if assignment.step_id == step_id:
                task.assignments[index] = assignment.model_copy(update={"status": status})
                return

    @staticmethod
    def _record_milestone(
        task: TaskState,
        code: MilestoneCode,
        title: str,
        status: MilestoneStatus,
        details: dict,
    ) -> None:
        milestone = TaskMilestone(
            code=code,
            title=title,
            status=status,
            details=details,
        )
        for index, existing in enumerate(task.milestones):
            if existing.code == code:
                task.milestones[index] = milestone
                return
        task.milestones.append(milestone)

    def _record_experience(
        self,
        task: TaskState,
        outcome: str,
        *,
        failure_type: Optional[str] = None,
    ) -> None:
        if task.scratchpad.get("experience_recorded"):
            return
        project_id = self._project_id(
            task.request.repository,
            str(task.scratchpad.get("tenant_id", self.tenant_id)),
        )
        record = {
            "task_id": task.task_id,
            "outcome": outcome,
            "task_type": task.normalized.task_type.value if task.normalized else "unknown",
            "route": task.route.route.value if task.route else "unknown",
            "changed_files": list(task.changed_files),
            "verification": [
                {
                    "command": result.command,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                }
                for result in task.verification
            ],
            "repair_attempts": len(task.repair_history),
            "collaboration_rounds": len(task.collaboration_history),
            "repair_fingerprints": [plan.failure_fingerprint for plan in task.repair_history],
            "failure_type": failure_type,
            "retrieval_stop_reason": (
                task.retrieval_trace.stop_reason if task.retrieval_trace else None
            ),
        }
        self.memory.append_bounded(
            MemoryLayer.PROJECT,
            project_id,
            "experiences",
            record,
            max_items=20,
            unique_by="task_id",
        )
        task.scratchpad["experience_recorded"] = True

    def _workspace_repository(self, task: TaskState) -> Repository:
        return Repository(self._ensure_workspace(task))

    def _source_repository(self, task: TaskState) -> Repository:
        source = Path(task.request.repository)
        if source.is_dir():
            return Repository(source)
        snapshot_id = task.scratchpad.get("source_snapshot_id")
        if self.workspace_store is None or not isinstance(snapshot_id, str):
            raise FileNotFoundError("source repository and snapshot are unavailable")
        return Repository(self.workspace_store.source_path(snapshot_id))

    def _ensure_workspace(self, task: TaskState) -> Path:
        workspace = Path(str(task.scratchpad.get("workspace", "")))
        if workspace.is_dir():
            return workspace
        snapshot_id = task.scratchpad.get("task_workspace_snapshot_id")
        if self.workspace_store is None or not isinstance(snapshot_id, str):
            raise FileNotFoundError("task workspace is unavailable")
        destination = self.data_directory / "workspaces" / task.task_id
        restored = self.workspace_store.restore(snapshot_id, destination)
        task.scratchpad["workspace"] = str(restored)
        return restored

    def publish_workspace(self, task_id: str) -> None:
        if self.workspace_store is None:
            return
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        snapshot_id = state.scratchpad.get("task_workspace_snapshot_id")
        if not isinstance(snapshot_id, str):
            raise RuntimeError("task workspace snapshot id is unavailable")
        workspace = self._ensure_workspace(state)
        self.workspace_store.replace(snapshot_id, workspace)

    def _create_workspace(self, task_id: str, source: Path) -> Path:
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = self.data_directory / "workspaces" / task_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            ignore=shutil.ignore_patterns(*IGNORED_WORKSPACE_NAMES),
        )
        return destination

    @staticmethod
    def _project_id(repository: str, tenant_id: str = "local") -> str:
        identity = f"{tenant_id}:{Path(repository).resolve()}"
        return sha256(identity.encode()).hexdigest()[:20]

    def _should_collaborate(self, route: RouteKind) -> bool:
        if self.collaboration_mode == "forced":
            return True
        if self.collaboration_mode == "disabled":
            return False
        return route in {RouteKind.COMPLEX, RouteKind.HUMAN_GATE}

    @staticmethod
    def _result(state: TaskState) -> TaskResult:
        return TaskResult(
            task_id=state.task_id,
            status=state.status,
            normalized=state.normalized,
            route=state.route,
            milestones=state.milestones,
            changed_files=state.changed_files,
            verification=state.verification,
            review=state.review,
            repair_history=state.repair_history,
            collaboration_history=state.collaboration_history,
            approval=state.approval,
            model_usage=state.model_usage,
            retrieval_trace=state.retrieval_trace,
            trace=state.trace,
            error=state.error,
        )
