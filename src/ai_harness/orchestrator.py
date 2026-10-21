from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import AgentModel, CoderAgent, ReviewerAgent
from .checkpoints import CheckpointStore
from .models import TaskRequest, TaskResult, TaskState, TaskStatus
from .planner import TaskRouter
from .repository import ContextIndex, RepoMapBuilder, Repository
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
    ) -> None:
        self.model = model
        self.data_directory = data_directory.resolve()
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints = checkpoint_store or CheckpointStore(
            self.data_directory / "checkpoints.sqlite3"
        )
        self.router = TaskRouter()
        self.repo_map_builder = RepoMapBuilder()
        self.coder = CoderAgent(model)
        self.reviewer = ReviewerAgent()
        self.tools = default_registry()
        self.runner_factory = runner_factory or (lambda workspace: DockerSandbox(workspace))

    def run(self, request: TaskRequest) -> TaskResult:
        task_id = str(uuid.uuid4())
        workspace = self._create_workspace(task_id, Path(request.repository))
        state = TaskState(
            task_id=task_id,
            request=request,
            status=TaskStatus.RUNNING,
            plan=self.router.build_plan(request),
            scratchpad={"workspace": str(workspace)},
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

    def _invoke(self, state: TaskState) -> TaskResult:
        graph = self._build_graph()
        try:
            output = graph.invoke({"task": state})
            final_state = output["task"]
        except Exception as exc:
            state.status = TaskStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            state.current_node = "failed"
            self.checkpoints.save(state)
            final_state = state
        return self._result(final_state)

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("scout", self._scout)
        graph.add_node("code", self._code)
        graph.add_node("verify", self._verify)
        graph.add_node("review", self._review)
        graph.add_conditional_edges(
            START,
            self._resume_route,
            {"scout": "scout", "code": "code", "verify": "verify", "review": "review", "end": END},
        )
        graph.add_edge("scout", "code")
        graph.add_edge("code", "verify")
        graph.add_edge("verify", "review")
        graph.add_conditional_edges(
            "review",
            self._review_route,
            {"code": "code", "end": END},
        )
        return graph.compile()

    @staticmethod
    def _resume_route(
        payload: WorkflowState,
    ) -> Literal["scout", "code", "verify", "review", "end"]:
        task = payload["task"]
        current = task.current_node
        if current == "review" and task.status == TaskStatus.RUNNING:
            return "code"
        return {
            "start": "scout",
            "scout": "code",
            "code": "verify",
            "verify": "review",
        }.get(current, "end")  # type: ignore[return-value]

    @staticmethod
    def _review_route(payload: WorkflowState) -> Literal["code", "end"]:
        task = payload["task"]
        if task.review and task.review.approved:
            return "end"
        if task.status == TaskStatus.RUNNING:
            return "code"
        return "end"

    def _scout(self, payload: WorkflowState) -> WorkflowState:
        task = payload["task"]
        repository = self._workspace_repository(task)
        task.repo_map = self.repo_map_builder.build(repository, task.request.file_scope)
        task.context = ContextIndex(repository, task.repo_map).search(task.request.goal, limit=30)
        task.current_node = "scout"
        self.checkpoints.save(task)
        return {"task": task}

    def _code(self, payload: WorkflowState) -> WorkflowState:
        task = payload["task"]
        previous_failure = "\n".join(
            result.stderr or result.stdout for result in task.verification if result.exit_code != 0
        )[-4000:]
        decision = self.coder.propose(task.request, task.context, previous_failure)
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
        task.current_node = "code"
        self.checkpoints.save(task)
        return {"task": task}

    def _verify(self, payload: WorkflowState) -> WorkflowState:
        task = payload["task"]
        workspace = Path(str(task.scratchpad["workspace"]))
        sandbox = self.runner_factory(workspace)
        task.verification = [sandbox.run(command) for command in task.request.verification_commands]
        task.current_node = "verify"
        self.checkpoints.save(task)
        return {"task": task}

    def _review(self, payload: WorkflowState) -> WorkflowState:
        task = payload["task"]
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
        task.current_node = "review"
        self.checkpoints.save(task)
        return {"task": task}

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
    def _result(state: TaskState) -> TaskResult:
        return TaskResult(
            task_id=state.task_id,
            status=state.status,
            changed_files=state.changed_files,
            verification=state.verification,
            review=state.review,
            error=state.error,
        )
