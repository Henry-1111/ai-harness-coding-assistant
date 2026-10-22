from __future__ import annotations

import shutil
import time
import uuid
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import AgentModel, CoderAgent, ReviewerAgent
from .checkpoints import CheckpointStore
from .embeddings import EmbeddingProvider
from .git_tools import GitCommandError, GitRepository
from .knowledge import (
    KnowledgeIdentity,
    RestrictedKnowledgeAgent,
    repository_knowledge_registry,
    sources_to_hits,
)
from .memory import LayeredMemoryStore, MemoryLayer
from .models import (
    AgentTraceEvent,
    TaskMilestone,
    TaskRequest,
    TaskResult,
    TaskState,
    TaskStatus,
)
from .planner import TaskRouter
from .repository import (
    DefinitionProvider,
    ProgressiveRetriever,
    RepoMapBuilder,
    Repository,
    RetrievalCache,
)
from .sandbox import CommandRunner, DockerSandbox
from .tools import ToolContext, default_registry


class WorkflowState(TypedDict):
    task: TaskState


class HarnessOrchestrator:
    """LangGraph-backed Scout -> Coder -> Verify -> Reviewer workflow."""

    def __init__(
        self,
        model: AgentModel,
        data_directory: Path,
        *,
        checkpoint_store: Optional[CheckpointStore] = None,
        runner_factory: Optional[Callable[[Path], CommandRunner]] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        definition_provider_factory: Optional[Callable[[Path], DefinitionProvider]] = None,
        restricted_react_enabled: bool = False,
        restricted_react_max_tool_calls: int = 3,
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
        self.memory = LayeredMemoryStore(self.data_directory / "memory.sqlite3")
        if self.memory.get(MemoryLayer.POLICY, "default", "instructions") is None:
            self.memory.put(
                MemoryLayer.POLICY,
                "default",
                "instructions",
                {"require_tests": True, "prefer_smallest_complete_change": True},
            )
        self.coder = CoderAgent(model)
        self.reviewer = ReviewerAgent()
        self.tools = default_registry()
        self.runner_factory = runner_factory or (lambda workspace: DockerSandbox(workspace))

    def run(self, request: TaskRequest) -> TaskResult:
        task_id = str(uuid.uuid4())
        workspace = self._create_workspace(task_id, Path(request.repository))
        normalized = self.router.normalizer.normalize(request)
        route = self.router.decide(normalized)
        plan = self.router.build_plan(request, normalized, route)
        state = TaskState(
            task_id=task_id,
            request=request,
            status=TaskStatus.RUNNING,
            normalized=normalized,
            route=route,
            plan=plan,
            assignments=self.router.assignments(plan),
            scratchpad={"workspace": str(workspace)},
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
        self.checkpoints.save(state)
        return self._invoke(state)

    def resume(self, task_id: str) -> TaskResult:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
            return self._result(state)
        workspace = Path(str(state.scratchpad.get("workspace", "")))
        if not workspace.is_dir():
            raise FileNotFoundError("task workspace is unavailable")
        state.status = TaskStatus.RUNNING
        state.error = None
        return self._invoke(state)

    def get(self, task_id: str) -> Optional[TaskState]:
        return self.checkpoints.load(task_id)

    def promote(self, task_id: str) -> list[str]:
        """Copy only reviewed files from the isolated workspace to the source repo."""
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if state.status != TaskStatus.SUCCEEDED or not state.review or not state.review.approved:
            raise RuntimeError("only approved tasks can be promoted")
        source = Repository(Path(state.request.repository))
        workspace = Repository(Path(str(state.scratchpad["workspace"])))
        promoted = []
        for relative_path in state.changed_files:
            source_path = workspace.resolve(relative_path)
            destination = source.resolve(relative_path, must_exist=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            promoted.append(relative_path)
        return promoted

    def diff(self, task_id: str) -> str:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        source = Repository(Path(state.request.repository))
        workspace = Repository(Path(str(state.scratchpad["workspace"])))
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

    def _invoke(self, state: TaskState) -> TaskResult:
        graph = self._build_graph()
        try:
            output = graph.invoke({"task": state})
            final_state = output["task"]
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
            self.checkpoints.save(state)
            final_state = state
        return self._result(final_state)

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

    def _scout(self, payload: WorkflowState) -> WorkflowState:
        started = time.monotonic()
        task = payload["task"]
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
        task.retrieval_trace = index.last_trace
        knowledge_result = None
        if self.restricted_react_enabled and index.last_trace.stop_reason != "evidence_sufficient":
            registry = repository_knowledge_registry(repository, task.repo_map)
            identity = KnowledgeIdentity(
                tenant_id="local",
                project_id=self._project_id(task.request.repository),
                allowed_tools=frozenset({"search_project_knowledge", "get_source_detail"}),
                allowed_paths=task.request.file_scope,
            )
            knowledge_result = RestrictedKnowledgeAgent(
                self.model,
                registry,
                max_tool_calls=self.restricted_react_max_tool_calls,
            ).run(task.request.goal, identity)
            known = {(hit.path, hit.line, hit.text) for hit in task.context}
            for hit in sources_to_hits(knowledge_result.sources):
                key = (hit.path, hit.line, hit.text)
                if key not in known:
                    task.context.append(hit)
                    known.add(key)
            task.scratchpad["restricted_react"] = knowledge_result.model_dump(mode="json")
        task.scratchpad["git_changed_paths"] = changed_paths
        project_id = self._project_id(task.request.repository)
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

    def _plan(self, payload: WorkflowState) -> WorkflowState:
        started = time.monotonic()
        task = payload["task"]
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

    def _code(self, payload: WorkflowState) -> WorkflowState:
        started = time.monotonic()
        task = payload["task"]
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
        memory_context = self.memory.resolve(
            "instructions",
            policy_id="default",
            project_id=self._project_id(task.request.repository),
            user_id="local",
            session_id=task.task_id,
        )
        decision = self.coder.propose(
            task.request,
            task.context,
            previous_failure,
            memory_context=memory_context,
        )
        repository = self._workspace_repository(task)
        context = ToolContext(
            repository=repository,
            permissions={"read", "write"},
            allowed_paths=task.request.file_scope,
        )
        for edit in decision.edits:
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
                details={"changed_files": list(task.changed_files)},
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

    def _verify(self, payload: WorkflowState) -> WorkflowState:
        started = time.monotonic()
        task = payload["task"]
        self._set_assignment_status(task, "verify", "running")
        workspace = Path(str(task.scratchpad["workspace"]))
        sandbox = self.runner_factory(workspace)
        task.verification = [sandbox.run(command) for command in task.request.verification_commands]
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

    def _review(self, payload: WorkflowState) -> WorkflowState:
        started = time.monotonic()
        task = payload["task"]
        self._set_assignment_status(task, "review", "running")
        task.review = self.reviewer.review(
            task.request,
            task.changed_files,
            [result.exit_code for result in task.verification],
        )
        if task.review.approved:
            task.status = TaskStatus.SUCCEEDED
        elif task.replan_count < task.request.max_replans:
            task.replan_count += 1
            task.status = TaskStatus.RUNNING
        else:
            task.status = TaskStatus.FAILED
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
        code: str,
        title: str,
        status: str,
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

    def _workspace_repository(self, task: TaskState) -> Repository:
        return Repository(Path(str(task.scratchpad["workspace"])))

    def _create_workspace(self, task_id: str, source: Path) -> Path:
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = self.data_directory / "workspaces" / task_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ai-harness"
            ),
        )
        return destination

    @staticmethod
    def _project_id(repository: str) -> str:
        return sha256(str(Path(repository).resolve()).encode()).hexdigest()[:20]

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
            retrieval_trace=state.retrieval_trace,
            trace=state.trace,
            error=state.error,
        )
