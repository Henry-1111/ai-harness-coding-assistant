from pathlib import Path

from ai_harness.memory import LayeredMemoryStore, MemoryLayer


def test_layered_memory_isolated_and_resolved_in_order(tmp_path: Path):
    store = LayeredMemoryStore(tmp_path / "memory.sqlite3")
    store.put(MemoryLayer.POLICY, "default", "style", {"tests": "required"})
    store.put(MemoryLayer.PROJECT, "project-a", "style", {"line_length": 100})
    store.put(MemoryLayer.USER, "user-a", "style", {"language": "zh"})
    store.put(MemoryLayer.PROJECT, "project-b", "style", {"line_length": 88})

    resolved = store.resolve(
        "style",
        policy_id="default",
        project_id="project-a",
        user_id="user-a",
        session_id="session-a",
    )

    assert [item["layer"] for item in resolved] == ["policy", "project", "user"]
    assert resolved[1]["value"] == {"line_length": 100}
    assert store.get(MemoryLayer.PROJECT, "project-b", "style") == {"line_length": 88}


def test_memory_upsert_replaces_same_scope_only(tmp_path: Path):
    store = LayeredMemoryStore(tmp_path / "memory.sqlite3")
    store.put(MemoryLayer.SESSION, "session-a", "progress", {"step": 1})
    store.put(MemoryLayer.SESSION, "session-a", "progress", {"step": 2})

    assert store.get(MemoryLayer.SESSION, "session-a", "progress") == {"step": 2}
    assert store.get(MemoryLayer.SESSION, "session-b", "progress") is None


def test_bounded_memory_appends_deduplicates_and_evicts_oldest(tmp_path: Path):
    store = LayeredMemoryStore(tmp_path / "memory.sqlite3")

    store.append_bounded(
        MemoryLayer.PROJECT,
        "project-a",
        "experiences",
        {"task_id": "one", "outcome": "failed"},
        max_items=2,
        unique_by="task_id",
    )
    store.append_bounded(
        MemoryLayer.PROJECT,
        "project-a",
        "experiences",
        {"task_id": "two", "outcome": "succeeded"},
        max_items=2,
        unique_by="task_id",
    )
    updated = store.append_bounded(
        MemoryLayer.PROJECT,
        "project-a",
        "experiences",
        {"task_id": "one", "outcome": "succeeded"},
        max_items=2,
        unique_by="task_id",
    )

    assert [item["task_id"] for item in updated] == ["two", "one"]
    assert updated[-1]["outcome"] == "succeeded"
