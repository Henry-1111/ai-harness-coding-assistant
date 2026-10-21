from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models import TaskRequest, TaskResult, TaskState
from .orchestrator import HarnessOrchestrator


def create_app(orchestrator: HarnessOrchestrator) -> FastAPI:
    app = FastAPI(title="AI Harness", version="0.1.0")
    app.state.orchestrator = orchestrator

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
