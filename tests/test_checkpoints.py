from pathlib import Path

from ai_harness.checkpoints import CheckpointStore
from ai_harness.models import TaskRequest, TaskState, TaskStatus


def test_checkpoint_store_keeps_revisions_and_loads_latest(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoints.sqlite3")
    state = TaskState(
        task_id="task-1",
        request=TaskRequest(repository=str(tmp_path), goal="Implement a useful feature"),
    )
    assert store.save(state) == 1
    state.status = TaskStatus.RUNNING
    state.current_node = "scout"
    assert store.save(state) == 2

    loaded = store.load("task-1")

    assert loaded is not None
    assert loaded.current_node == "scout"
    assert loaded.status == TaskStatus.RUNNING
    assert store.revisions("task-1") == [1, 2]


def test_checkpoint_revision_load_and_binary_search(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoints.sqlite3")
    state = TaskState(
        task_id="task-bisect",
        request=TaskRequest(repository=str(tmp_path), goal="Find first failed revision"),
    )
    for node, status in [
        ("scout", TaskStatus.RUNNING),
        ("code", TaskStatus.RUNNING),
        ("verify", TaskStatus.FAILED),
        ("review", TaskStatus.FAILED),
    ]:
        state.current_node = node
        state.status = status
        store.save(state)

    revision = store.bisect_first(
        "task-bisect", lambda item: item.status == TaskStatus.FAILED
    )

    assert revision == 3
    assert store.load_revision("task-bisect", revision).current_node == "verify"
