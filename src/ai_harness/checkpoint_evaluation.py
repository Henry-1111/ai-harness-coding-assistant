from __future__ import annotations

import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from .checkpoints import CheckpointStore
from .models import TaskRequest, TaskState, TaskStatus, utc_now
from .runtime_evaluation import LatencyPercentiles, percentile


class CheckpointBenchmarkReport(BaseModel):
    benchmark: str = "sqlite_checkpoint_recovery"
    generated_at: str = Field(default_factory=lambda: utc_now().isoformat())
    tasks: int
    revisions_per_task: int
    writers: int
    corruption_trials: int
    total_checkpoints: int
    recovered_tasks: int
    recovery_rate: float
    revision_integrity_rate: float
    corruption_isolation_rate: float
    recovery_latency_ms: LatencyPercentiles
    failed_task_ids: List[str] = Field(default_factory=list)


def run_checkpoint_benchmark(
    *,
    tasks: int = 200,
    revisions_per_task: int = 8,
    writers: int = 8,
    corruption_trials: int = 20,
) -> CheckpointBenchmarkReport:
    if not 1 <= tasks <= 5_000:
        raise ValueError("tasks must be within 1..5000")
    if not 1 <= revisions_per_task <= 50:
        raise ValueError("revisions_per_task must be within 1..50")
    if not 1 <= writers <= 32:
        raise ValueError("writers must be within 1..32")
    if not 0 <= corruption_trials <= 100:
        raise ValueError("corruption_trials must be within 0..100")

    nodes = ("normalize", "scout", "plan", "code", "verify", "review")
    with tempfile.TemporaryDirectory(prefix="ai-harness-checkpoint-benchmark-") as directory:
        database_path = Path(directory) / "checkpoints.sqlite3"
        store = CheckpointStore(database_path)

        def write_task(index: int) -> str:
            task_id = f"checkpoint-task-{index:05d}"
            state = TaskState(
                task_id=task_id,
                request=TaskRequest(
                    repository=directory,
                    goal=f"Recover interrupted task {index}",
                    verification_commands=[],
                ),
                status=TaskStatus.RUNNING,
            )
            for revision in range(1, revisions_per_task + 1):
                state.current_node = nodes[(revision - 1) % len(nodes)]
                state.scratchpad = {
                    "sequence": revision,
                    "payload": f"task-{index}-revision-{revision}",
                }
                saved_revision = store.save(state)
                if saved_revision != revision:
                    raise RuntimeError(
                        f"unexpected revision for {task_id}: {saved_revision}"
                    )
            return task_id

        with ThreadPoolExecutor(max_workers=writers) as executor:
            task_ids = list(executor.map(write_task, range(tasks)))

        # Constructing a new store simulates a fresh process opening the durable file.
        recovered_store = CheckpointStore(database_path)
        recovered_tasks = 0
        histories_valid = 0
        recovery_latencies: List[float] = []
        failed_task_ids: List[str] = []
        expected_revisions = list(range(1, revisions_per_task + 1))
        for task_id in task_ids:
            started = time.perf_counter()
            latest = recovered_store.load(task_id)
            recovery_latencies.append((time.perf_counter() - started) * 1000)
            latest_valid = (
                latest is not None
                and latest.status == TaskStatus.RUNNING
                and latest.scratchpad.get("sequence") == revisions_per_task
                and latest.scratchpad.get("payload")
                == f"task-{int(task_id.rsplit('-', 1)[1])}-revision-{revisions_per_task}"
            )
            if latest_valid:
                recovered_tasks += 1
            else:
                failed_task_ids.append(task_id)

            revisions = recovered_store.revisions(task_id)
            history_valid = revisions == expected_revisions
            if history_valid:
                for revision in revisions:
                    historical = recovered_store.load_revision(task_id, revision)
                    if historical is None or historical.scratchpad.get("sequence") != revision:
                        history_valid = False
                        break
            histories_valid += int(history_valid)
            if not history_valid and task_id not in failed_task_ids:
                failed_task_ids.append(task_id)

        isolated_corruptions = 0
        healthy_task_id = task_ids[0]
        with sqlite3.connect(str(database_path)) as connection:
            for index in range(corruption_trials):
                connection.execute(
                    "INSERT INTO task_checkpoints "
                    "(task_id, revision, current_node, status, state_json, created_at) "
                    "VALUES (?, 1, 'code', 'running', ?, ?)",
                    (f"corrupt-task-{index}", "{invalid-json", utc_now().isoformat()),
                )
        for index in range(corruption_trials):
            corrupt_failed_closed = False
            try:
                recovered_store.load(f"corrupt-task-{index}")
            except (ValueError, TypeError):
                corrupt_failed_closed = True
            healthy = recovered_store.load(healthy_task_id)
            isolated_corruptions += int(corrupt_failed_closed and healthy is not None)

        return CheckpointBenchmarkReport(
            tasks=tasks,
            revisions_per_task=revisions_per_task,
            writers=writers,
            corruption_trials=corruption_trials,
            total_checkpoints=tasks * revisions_per_task,
            recovered_tasks=recovered_tasks,
            recovery_rate=round(recovered_tasks / tasks, 6),
            revision_integrity_rate=round(histories_valid / tasks, 6),
            corruption_isolation_rate=(
                round(isolated_corruptions / corruption_trials, 6)
                if corruption_trials
                else 1.0
            ),
            recovery_latency_ms=LatencyPercentiles(
                p50=round(percentile(recovery_latencies, 0.50), 3),
                p95=round(percentile(recovery_latencies, 0.95), 3),
                p99=round(percentile(recovery_latencies, 0.99), 3),
            ),
            failed_task_ids=failed_task_ids,
        )
