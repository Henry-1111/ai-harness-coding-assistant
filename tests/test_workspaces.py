from pathlib import Path

import pytest

from ai_harness.workspaces import (
    FilesystemWorkspaceSnapshotStore,
    WorkspaceSnapshotCapacityError,
)


def test_workspace_snapshot_is_atomic_idempotent_and_excludes_heavy_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("print('safe')\n")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("secret")
    store = FilesystemWorkspaceSnapshotStore(tmp_path / "shared")

    assert store.healthcheck() is True
    assert store.stage("job-123", source) is True
    assert store.stage("job-123", source) is False

    snapshot = store.source_path("job-123")
    assert (snapshot / "app.py").read_text() == "print('safe')\n"
    assert not (snapshot / ".git").exists()


def test_workspace_snapshot_preserves_symlinks_without_copying_external_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "outside.txt"
    external.write_text("outside")
    (source / "link.txt").symlink_to(external)
    store = FilesystemWorkspaceSnapshotStore(tmp_path / "shared")

    store.stage("job-123", source)

    assert (store.source_path("job-123") / "link.txt").is_symlink()


def test_workspace_snapshot_can_replace_and_restore_task_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "app.py"
    tracked.write_text("version one\n")
    store = FilesystemWorkspaceSnapshotStore(tmp_path / "shared")
    store.replace("task-123", source)

    tracked.write_text("version two\n")
    store.replace("task-123", source)
    restored = store.restore("task-123", tmp_path / "other-host" / "workspace")

    assert (restored / "app.py").read_text() == "version two\n"


@pytest.mark.parametrize("snapshot_id", ["", "../escape", "slash/value", "x" * 129])
def test_workspace_snapshot_rejects_unsafe_ids(
    tmp_path: Path, snapshot_id: str
) -> None:
    store = FilesystemWorkspaceSnapshotStore(tmp_path / "shared")

    with pytest.raises(ValueError, match="invalid workspace snapshot id"):
        store.source_path(snapshot_id)


def test_workspace_snapshot_rejects_storage_nested_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = FilesystemWorkspaceSnapshotStore(source / ".snapshots")

    with pytest.raises(ValueError, match="cannot be inside source"):
        store.stage("job-123", source)


def test_workspace_snapshot_enforces_total_capacity_and_reports_usage(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    (first / "data.bin").write_bytes(b"a" * 6)
    second = tmp_path / "second"
    second.mkdir()
    (second / "data.bin").write_bytes(b"b" * 5)
    store = FilesystemWorkspaceSnapshotStore(tmp_path / "shared", max_bytes=10)

    store.stage("first", first)

    with pytest.raises(WorkspaceSnapshotCapacityError, match="4 bytes available"):
        store.stage("second", second)

    assert store.metrics().snapshot_count == 1
    assert store.metrics().bytes_used == 6
    assert store.metrics().max_bytes == 10


def test_failed_capacity_checked_replace_preserves_existing_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "app.py"
    tracked.write_text("small")
    store = FilesystemWorkspaceSnapshotStore(tmp_path / "shared", max_bytes=8)
    store.replace("task-123", source)
    tracked.write_text("too large for configured capacity")

    with pytest.raises(WorkspaceSnapshotCapacityError):
        store.replace("task-123", source)

    assert (store.source_path("task-123") / "app.py").read_text() == "small"


def test_workspace_snapshot_capacity_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        FilesystemWorkspaceSnapshotStore(tmp_path / "shared", max_bytes=0)
