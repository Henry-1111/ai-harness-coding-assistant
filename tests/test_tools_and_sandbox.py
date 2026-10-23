from pathlib import Path
from subprocess import TimeoutExpired

import pytest

from ai_harness.models import FileEdit
from ai_harness.repository import Repository
from ai_harness.sandbox import (
    DockerSandbox,
    LocalCommandSandbox,
    SandboxViolation,
    validate_command,
)
from ai_harness.tools import (
    ToolContext,
    ToolExecutionError,
    ToolPermissionError,
    apply_edit,
    default_registry,
)


def test_apply_edit_requires_unique_old_text(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    with pytest.raises(ToolExecutionError, match="exactly one"):
        apply_edit(context, FileEdit(path="app.py", old_text="value = 1", new_text="value = 2"))


def test_apply_edit_enforces_scope(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, ["src"])

    with pytest.raises(ToolPermissionError, match="outside task scope"):
        apply_edit(context, FileEdit(path="app.py", old_text="value = 1", new_text="value = 2"))


def test_apply_edit_can_create_scoped_file(tmp_path: Path):
    context = ToolContext(Repository(tmp_path), {"write"}, ["src"])
    path = apply_edit(context, FileEdit(path="src/new.py", old_text="", new_text="answer = 42\n"))

    assert path == "src/new.py"
    assert (tmp_path / path).read_text(encoding="utf-8") == "answer = 42\n"


def test_registry_checks_tool_permission(tmp_path: Path):
    (tmp_path / "app.py").write_text("answer = 41\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"read"}, [])

    with pytest.raises(ToolPermissionError, match="requires permission write"):
        default_registry().execute(
            "apply_edit",
            context,
            edit=FileEdit(path="app.py", old_text="41", new_text="42"),
        )


def test_command_validation_rejects_shell(tmp_path: Path):
    with pytest.raises(SandboxViolation, match="not allow-listed"):
        validate_command("sh -c 'rm -rf .'", {"python"})


def test_local_command_sandbox_runs_without_shell(tmp_path: Path):
    result = LocalCommandSandbox(tmp_path).run("python -c 'print(6 * 7)'")

    assert result.exit_code == 0
    assert result.stdout.strip() == "42"


def test_local_command_sandbox_preserves_timeout_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def time_out(*_args: object, **_kwargs: object) -> None:
        raise TimeoutExpired(
            cmd=["python"], timeout=1, output=b"partial output", stderr=b"timed out"
        )

    monkeypatch.setattr("ai_harness.sandbox.subprocess.run", time_out)

    result = LocalCommandSandbox(tmp_path, timeout_seconds=1).run("python -V")

    assert result.exit_code == 124
    assert result.timed_out is True
    assert result.stdout == "partial output"
    assert result.stderr == "timed out"


def test_docker_sandbox_applies_resource_and_network_limits(tmp_path: Path):
    arguments = DockerSandbox(tmp_path).docker_arguments("python -m pytest -q")

    assert "none" in arguments
    assert "--memory" in arguments
    assert "--cpus" in arguments
    assert "no-new-privileges" in arguments
    assert arguments[-4:] == ["python", "-m", "pytest", "-q"]
