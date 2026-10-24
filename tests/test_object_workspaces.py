from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from ai_harness.object_workspaces import S3WorkspaceSnapshotStore


class S3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str], str]] = {}
        self.put_error: Exception | None = None

    def head_bucket(self, **_: Any) -> dict[str, Any]:
        return {}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.put_error is not None:
            raise self.put_error
        key = str(kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise S3Error("PreconditionFailed")
        data = kwargs["Body"].read()
        etag = hashlib.sha256(data).hexdigest()
        self.objects[key] = (data, dict(kwargs.get("Metadata", {})), etag)
        return {"ETag": etag}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        try:
            data, metadata, etag = self.objects[str(kwargs["Key"])]
        except KeyError as exc:
            raise S3Error("NoSuchKey") from exc
        return {"ContentLength": len(data), "Metadata": metadata, "ETag": etag}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        data, metadata, etag = self.objects[str(kwargs["Key"])]
        return {"Body": io.BytesIO(data), "Metadata": metadata, "ETag": etag}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.objects.pop(str(kwargs["Key"]), None)
        return {}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = str(kwargs.get("Prefix", ""))
        contents = [
            {"Key": key, "Size": len(data)}
            for key, (data, _, _) in self.objects.items()
            if key.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}


def test_s3_snapshot_round_trip_and_cross_host_cache_refresh(tmp_path: Path) -> None:
    client = FakeS3Client()
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "app.py"
    tracked.write_text("version one\n")
    external = tmp_path / "external.txt"
    external.write_text("outside\n")
    (source / "external-link").symlink_to(external)
    (source / "internal-link").symlink_to("app.py")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("must not be archived")
    first = S3WorkspaceSnapshotStore(
        tmp_path / "cache-one", bucket="snapshots", client=client
    )
    second = S3WorkspaceSnapshotStore(
        tmp_path / "cache-two", bucket="snapshots", client=client
    )

    assert first.stage("job-123", source) is True
    assert second.stage("job-123", source) is False
    assert (second.source_path("job-123") / "app.py").read_text() == "version one\n"
    assert not (second.source_path("job-123") / "external-link").is_symlink()
    assert (second.source_path("job-123") / "internal-link").is_symlink()
    assert not (second.source_path("job-123") / ".git").exists()

    tracked.write_text("version two\n")
    first.replace("job-123", source)

    assert (second.source_path("job-123") / "app.py").read_text() == "version two\n"
    restored = second.restore("job-123", tmp_path / "restored")
    assert (restored / "app.py").read_text() == "version two\n"
    assert second.metrics().snapshot_count == 1
    assert second.metrics().bytes_used > 0
    assert second.healthcheck() is True


def test_s3_snapshot_rejects_checksum_mismatch_and_removes_remote_object(
    tmp_path: Path,
) -> None:
    client = FakeS3Client()
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("safe\n")
    store = S3WorkspaceSnapshotStore(
        tmp_path / "cache", bucket="snapshots", client=client
    )
    store.stage("job-123", source)
    key = "ai-harness/workspaces/job-123.tar.gz"
    _, metadata, _ = client.objects[key]
    client.objects[key] = (b"tampered", metadata, "different-etag")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        S3WorkspaceSnapshotStore(
            tmp_path / "other-cache", bucket="snapshots", client=client
        ).source_path("job-123")

    store.remove("job-123")
    assert key not in client.objects


def test_s3_snapshot_rejects_archive_path_traversal(tmp_path: Path) -> None:
    client = FakeS3Client()
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_data = archive_buffer.getvalue()
    digest = hashlib.sha256(archive_data).hexdigest()
    key = "ai-harness/workspaces/unsafe.tar.gz"
    client.objects[key] = (archive_data, {"sha256": digest}, "unsafe-etag")
    store = S3WorkspaceSnapshotStore(
        tmp_path / "cache", bucket="snapshots", client=client
    )

    with pytest.raises(RuntimeError, match="unsafe workspace snapshot archive path"):
        store.source_path("unsafe")

    assert not (tmp_path / "outside.txt").exists()


def test_s3_snapshot_rejects_escaping_symlink_target(tmp_path: Path) -> None:
    client = FakeS3Client()
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("nested/escape")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside.txt"
        archive.addfile(member)
    archive_data = archive_buffer.getvalue()
    digest = hashlib.sha256(archive_data).hexdigest()
    key = "ai-harness/workspaces/unsafe-link.tar.gz"
    client.objects[key] = (archive_data, {"sha256": digest}, "unsafe-link-etag")
    store = S3WorkspaceSnapshotStore(
        tmp_path / "cache", bucket="snapshots", client=client
    )

    with pytest.raises(RuntimeError, match="unsafe workspace snapshot symlink target"):
        store.source_path("unsafe-link")

    assert not (tmp_path / "outside.txt").exists()


def test_s3_cache_evicts_lru_entry_without_deleting_remote_snapshot(
    tmp_path: Path,
) -> None:
    client = FakeS3Client()
    first_source = tmp_path / "first-source"
    first_source.mkdir()
    (first_source / "payload.txt").write_text("a" * 1024)
    second_source = tmp_path / "second-source"
    second_source.mkdir()
    (second_source / "payload.txt").write_text("b" * 1024)
    third_source = tmp_path / "third-source"
    third_source.mkdir()
    (third_source / "payload.txt").write_text("c" * 1024)
    store = S3WorkspaceSnapshotStore(
        tmp_path / "cache",
        bucket="snapshots",
        max_bytes=2500,
        client=client,
    )

    assert store.stage("first", first_source) is True
    assert store.stage("second", second_source) is True
    os.utime(store.root / "first", ns=(1_000_000_000, 1_000_000_000))
    os.utime(store.root / "second", ns=(2_000_000_000, 2_000_000_000))
    store.source_path("first")
    assert store.stage("third", third_source) is True

    assert (store.root / "first").is_dir()
    assert not (store.root / "second").exists()
    assert (store.root / "third").is_dir()
    assert len(client.objects) == 3
    assert store.metrics().snapshot_count == 2
    assert store.metrics().bytes_used == 2048

    restored_second = store.source_path("second")

    assert (restored_second / "payload.txt").read_text() == "b" * 1024
    assert (store.root / "first").is_dir()
    assert not (store.root / "third").exists()
    assert len(client.objects) == 3


def test_failed_s3_replace_discards_uncommitted_cache_bytes(tmp_path: Path) -> None:
    client = FakeS3Client()
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "payload.txt"
    payload.write_text("remote version")
    store = S3WorkspaceSnapshotStore(
        tmp_path / "cache", bucket="snapshots", client=client
    )
    store.stage("snapshot", source)
    payload.write_text("uncommitted local version")
    client.put_error = S3Error("ServiceUnavailable")

    with pytest.raises(S3Error, match="ServiceUnavailable"):
        store.replace("snapshot", source)

    assert not (store.root / "snapshot").exists()
    client.put_error = None
    assert (store.source_path("snapshot") / "payload.txt").read_text() == "remote version"
