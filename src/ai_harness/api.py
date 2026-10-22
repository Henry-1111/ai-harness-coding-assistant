from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import TaskRequest, TaskResult, TaskState
from .orchestrator import HarnessOrchestrator


def create_app(orchestrator: HarnessOrchestrator) -> FastAPI:
    app = FastAPI(title="AI Harness", version="0.1.0")
    app.state.orchestrator = orchestrator
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/tasks", response_model=TaskResult)
    def create_task(request: TaskRequest) -> TaskResult:
        return orchestrator.run(request)

    @app.get("/v1/tasks/{task_id}", response_model=TaskState)
    def get_task(task_id: str) -> TaskState:
        task = orchestrator.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.post("/v1/tasks/{task_id}/resume", response_model=TaskResult)
    def resume_task(task_id: str) -> TaskResult:
        try:
            return orchestrator.resume(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/v1/tasks/{task_id}/diff")
    def task_diff(task_id: str) -> dict[str, str]:
        try:
            return {"task_id": task_id, "diff": orchestrator.diff(task_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/v1/tasks/{task_id}/trace")
    def task_trace(task_id: str) -> dict[str, object]:
        task = orchestrator.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "task_id": task_id,
            "events": [event.model_dump(mode="json") for event in task.trace],
            "retrieval": (
                task.retrieval_trace.model_dump(mode="json") if task.retrieval_trace else None
            ),
            "restricted_react": task.scratchpad.get("restricted_react"),
            "repairs": [item.model_dump(mode="json") for item in task.repair_history],
            "checkpoint_revisions": orchestrator.checkpoints.revisions(task_id),
        }

    @app.post("/v1/tasks/{task_id}/promote")
    def promote_task(task_id: str) -> dict[str, object]:
        try:
            files = orchestrator.promote(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"task_id": task_id, "promoted_files": files}

    return app
