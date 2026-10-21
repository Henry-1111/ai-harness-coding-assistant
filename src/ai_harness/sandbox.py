from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

from .models import CommandResult

DEFAULT_COMMAND_ALLOWLIST = {
    "pytest",
    "python",
    "python3",
    "ruff",
    "mypy",
    "npm",
    "pnpm",
    "yarn",
    "go",
    "cargo",
}


class SandboxViolation(PermissionError):
    pass


class CommandRunner(Protocol):
    def run(self, command: str) -> CommandResult: ...


def validate_command(command: str, allowlist: Iterable[str]) -> List[str]:
    arguments = shlex.split(command)
    if not arguments:
        raise SandboxViolation("empty command")
    executable = Path(arguments[0]).name
    if executable not in set(allowlist):
        raise SandboxViolation(f"command is not allow-listed: {executable}")
    return arguments


class LocalWorkspaceSandbox:
    """Disposable repository copy for deterministic tests and local development."""

    def __init__(
        self,
        source: Path,
        *,
        command_allowlist: Optional[Iterable[str]] = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.source = source.resolve()
        self.command_allowlist = set(command_allowlist or DEFAULT_COMMAND_ALLOWLIST)
        self.timeout_seconds = timeout_seconds
        self._temporary_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self.path: Optional[Path] = None

    def __enter__(self) -> "LocalWorkspaceSandbox":
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="ai-harness-")
        destination = Path(self._temporary_directory.name) / "workspace"
        shutil.copytree(
            self.source,
            destination,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"
            ),
        )
        self.path = destination
        return self

    def __exit__(self, *_: object) -> None:
        if self._temporary_directory:
            self._temporary_directory.cleanup()
        self.path = None

    def run(self, command: str) -> CommandResult:
        if self.path is None:
            raise RuntimeError("sandbox is not active")
        arguments = validate_command(command, self.command_allowlist)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                arguments,
                cwd=self.path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HOME": str(self.path / ".sandbox-home"),
                },
            )
            return CommandResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout[-20_000:],
                stderr=completed.stderr[-20_000:],
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=(exc.stdout or "")[-20_000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-20_000:] if isinstance(exc.stderr, str) else "",
                duration_ms=round((time.monotonic() - started) * 1000),
                timed_out=True,
            )


class LocalCommandSandbox:
    """Run allow-listed commands in an already isolated persistent workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        command_allowlist: Optional[Iterable[str]] = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.workspace = workspace.resolve()
        self.command_allowlist = set(command_allowlist or DEFAULT_COMMAND_ALLOWLIST)
        self.timeout_seconds = timeout_seconds

    def run(self, command: str) -> CommandResult:
        arguments = validate_command(command, self.command_allowlist)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                arguments,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HOME": str(self.workspace / ".sandbox-home"),
                },
            )
            return CommandResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout[-20_000:],
                stderr=completed.stderr[-20_000:],
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=(exc.stdout or "")[-20_000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-20_000:] if isinstance(exc.stderr, str) else "",
                duration_ms=round((time.monotonic() - started) * 1000),
                timed_out=True,
            )


class DockerSandbox:
    """Build a network-isolated, resource-bounded Docker command invocation."""

    def __init__(
        self,
        workspace: Path,
        image: str = "python:3.12-slim",
        timeout_seconds: int = 120,
        memory: str = "512m",
        cpus: str = "1.0",
        command_allowlist: Optional[Iterable[str]] = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory = memory
        self.cpus = cpus
        self.command_allowlist = set(command_allowlist or DEFAULT_COMMAND_ALLOWLIST)

    def docker_arguments(self, command: str) -> List[str]:
        arguments = validate_command(command, self.command_allowlist)
        container_name = f"ai-harness-{uuid.uuid4().hex[:12]}"
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:size=256m",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            "128",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "-e",
            "HOME=/tmp",
            "-v",
            f"{self.workspace}:/workspace",
            "-w",
            "/workspace",
            self.image,
            *arguments,
        ]

    def run(self, command: str) -> CommandResult:
        started = time.monotonic()
        docker_arguments = self.docker_arguments(command)
        container_name = docker_arguments[docker_arguments.index("--name") + 1]
        try:
            completed = subprocess.run(
                docker_arguments,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return CommandResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout[-20_000:],
                stderr=completed.stderr[-20_000:],
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return CommandResult(
                command=command,
                exit_code=124,
                duration_ms=round((time.monotonic() - started) * 1000),
                timed_out=True,
            )
