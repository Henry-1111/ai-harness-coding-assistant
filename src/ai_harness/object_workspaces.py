from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import tarfile
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from .workspaces import (
    DEFAULT_SNAPSHOT_CAPACITY_BYTES,
    IGNORED_WORKSPACE_NAMES,
    FilesystemWorkspaceSnapshotStore,
    WorkspaceSnapshotCapacityError,
    WorkspaceSnapshotMetrics,
    validate_snapshot_id,
)


class S3WorkspaceSnapshotStore:
    """S3-compatible durable snapshots backed by a bounded local cache."""

    def __init__(
        self,
        cache_root: Path,
        *,
        bucket: str,
        prefix: str = "ai-harness/workspaces",
        endpoint_url: str = "",
        region: str = "",
        access_key_id: str = "",
        secret_access_key: str = "",
        max_bytes: int = DEFAULT_SNAPSHOT_CAPACITY_BYTES,
        client: Any = None,
    ) -> None:
        if not bucket.strip():
            raise ValueError("workspace snapshot S3 bucket is required")
        self.bucket = bucket.strip()
        self.prefix = prefix.strip("/")
        self.object_prefix = f"{self.prefix}/" if self.prefix else ""
        self.cache = FilesystemWorkspaceSnapshotStore(cache_root, max_bytes=max_bytes)
        self.root = self.cache.root
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._known_etags: dict[str, str] = {}
        if client is None:
            boto3 = importlib.import_module("boto3")
            kwargs = {
                "endpoint_url": endpoint_url or None,
                "region_name": region or None,
                "aws_access_key_id": access_key_id or None,
                "aws_secret_access_key": secret_access_key or None,
            }
            client = boto3.client("s3", **{key: value for key, value in kwargs.items() if value})
        self.client = client

    @property
    def storage_identity(self) -> str:
        return f"s3:{self.bucket}/{self.prefix}"

    def _key(self, snapshot_id: str) -> str:
        name = f"{validate_snapshot_id(snapshot_id)}.tar.gz"
        return f"{self.object_prefix}{name}"

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"} or isinstance(exc, FileNotFoundError)

    @staticmethod
    def _is_precondition_failed(exc: Exception) -> bool:
        response = getattr(exc, "response", {})
        return str(response.get("Error", {}).get("Code", "")) in {
            "409",
            "412",
            "ConditionalRequestConflict",
            "PreconditionFailed",
        }

    def _head(self, snapshot_id: str) -> Optional[dict[str, Any]]:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=self._key(snapshot_id))
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise

    @staticmethod
    def _archive_filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        parts = PurePosixPath(info.name).parts
        if any(part in IGNORED_WORKSPACE_NAMES for part in parts):
            return None
        return info

    def _create_archive(self, source: Path, destination: Path) -> str:
        with tarfile.open(destination, "w:gz", dereference=False) as archive:
            archive.add(source, arcname=".", filter=self._archive_filter)
        digest = hashlib.sha256()
        with destination.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _extract_archive(archive_path: Path, destination: Path) -> None:
        with tarfile.open(archive_path, "r:gz") as archive:
            symlinks: list[tarfile.TarInfo] = []
            for member in archive.getmembers():
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError("unsafe workspace snapshot archive path")
                target = destination.joinpath(*member_path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("invalid workspace snapshot archive file")
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    os.chmod(target, member.mode & 0o777)
                elif member.issym():
                    symlinks.append(member)
                else:
                    raise RuntimeError("unsafe workspace snapshot archive member")
            for member in symlinks:
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.parent.resolve().relative_to(destination.resolve())
                except ValueError as exc:
                    raise RuntimeError("unsafe nested workspace snapshot symlink") from exc
                target.symlink_to(member.linkname)

    def _put(self, snapshot_id: str, source: Path, *, create_only: bool) -> bool:
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        with tempfile.TemporaryDirectory(
            prefix=".object-upload-", dir=self.root
        ) as directory:
            archive_path = Path(directory) / "snapshot.tar.gz"
            digest = self._create_archive(source, archive_path)
            archive_size = archive_path.stat().st_size
            if archive_size > self.max_bytes:
                raise WorkspaceSnapshotCapacityError(
                    f"workspace snapshot archive requires {archive_size} bytes, "
                    f"capacity is {self.max_bytes} bytes"
                )
            with archive_path.open("rb") as body:
                kwargs: dict[str, Any] = {
                    "Bucket": self.bucket,
                    "Key": self._key(snapshot_id),
                    "Body": body,
                    "ContentType": "application/gzip",
                    "Metadata": {"sha256": digest},
                }
                if create_only:
                    kwargs["IfNoneMatch"] = "*"
                try:
                    response = self.client.put_object(**kwargs)
                except Exception as exc:
                    if create_only and self._is_precondition_failed(exc):
                        return False
                    raise
        self._known_etags[snapshot_id] = str(response.get("ETag", ""))
        return True

    def stage(self, snapshot_id: str, source: Path) -> bool:
        snapshot_id = validate_snapshot_id(snapshot_id)
        with self._lock:
            self.cache.replace(snapshot_id, source)
            try:
                created = self._put(snapshot_id, source, create_only=True)
            except Exception:
                self.cache.remove(snapshot_id)
                raise
            if not created:
                self.cache.remove(snapshot_id)
                self._known_etags.pop(snapshot_id, None)
            return created

    def replace(self, snapshot_id: str, source: Path) -> None:
        snapshot_id = validate_snapshot_id(snapshot_id)
        with self._lock:
            self.cache.replace(snapshot_id, source)
            self._put(snapshot_id, source, create_only=False)

    def _download(self, snapshot_id: str, head: dict[str, Any]) -> Path:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(snapshot_id))
        with tempfile.TemporaryDirectory(
            prefix=".object-download-", dir=self.root
        ) as directory:
            archive_path = Path(directory) / "snapshot.tar.gz"
            digest = hashlib.sha256()
            with archive_path.open("wb") as target:
                body = response["Body"]
                while chunk := body.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
            expected = str(response.get("Metadata", {}).get("sha256", ""))
            if expected and digest.hexdigest() != expected:
                raise RuntimeError("workspace snapshot archive checksum mismatch")
            extracted = Path(directory) / "extracted"
            extracted.mkdir()
            self._extract_archive(archive_path, extracted)
            self.cache.replace(snapshot_id, extracted)
        self._known_etags[snapshot_id] = str(head.get("ETag", ""))
        return self.cache.source_path(snapshot_id)

    def source_path(self, snapshot_id: str) -> Path:
        snapshot_id = validate_snapshot_id(snapshot_id)
        with self._lock:
            head = self._head(snapshot_id)
            if head is None:
                raise FileNotFoundError(f"workspace snapshot is unavailable: {snapshot_id}")
            etag = str(head.get("ETag", ""))
            try:
                cached = self.cache.source_path(snapshot_id)
            except FileNotFoundError:
                cached = None
            if cached is not None and self._known_etags.get(snapshot_id) == etag:
                return cached
            return self._download(snapshot_id, head)

    def restore(self, snapshot_id: str, destination: Path) -> Path:
        self.source_path(snapshot_id)
        return self.cache.restore(snapshot_id, destination)

    def remove(self, snapshot_id: str) -> None:
        snapshot_id = validate_snapshot_id(snapshot_id)
        with self._lock:
            self.client.delete_object(Bucket=self.bucket, Key=self._key(snapshot_id))
            self.cache.remove(snapshot_id)
            self._known_etags.pop(snapshot_id, None)

    def healthcheck(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return self.cache.healthcheck()
        except Exception:
            return False

    def metrics(self) -> WorkspaceSnapshotMetrics:
        count = 0
        used = 0
        continuation: Optional[str] = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": self.object_prefix,
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation
            page = self.client.list_objects_v2(**kwargs)
            objects = [
                item
                for item in page.get("Contents", [])
                if str(item.get("Key", "")).endswith(".tar.gz")
            ]
            count += len(objects)
            used += sum(int(item.get("Size", 0)) for item in objects)
            if not page.get("IsTruncated"):
                break
            continuation = str(page["NextContinuationToken"])
        return WorkspaceSnapshotMetrics(count, used, self.max_bytes)
