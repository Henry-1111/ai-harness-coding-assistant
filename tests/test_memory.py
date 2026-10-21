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
