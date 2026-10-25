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
    apply_edits,
    default_registry,
)


def test_apply_edit_requires_unique_old_text(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    with pytest.raises(ToolExecutionError, match="exactly one"):
        apply_edit(context, FileEdit(path="app.py", old_text="value = 1", new_text="value = 2"))


def test_apply_edit_relocates_unique_numbered_retrieval_snippet(tmp_path: Path):
    (tmp_path / "app.py").write_text(
        "def answer():\n    return 41\n", encoding="utf-8"
    )
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    apply_edit(
        context,
        FileEdit(
            path="app.py",
            old_text="10: def answer():\n11:     return 41\n",
            new_text="10: def answer():\n11:     return 42\n",
        ),
    )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == (
        "def answer():\n    return 42\n"
    )


def test_apply_edit_rejects_ambiguous_numbered_retrieval_snippet(tmp_path: Path):
    (tmp_path / "app.py").write_text(
        "value = 1\nvalue = 1\n", encoding="utf-8"
    )
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    with pytest.raises(ToolExecutionError, match="exactly one"):
        apply_edit(
            context,
            FileEdit(path="app.py", old_text="7: value = 1", new_text="value = 2"),
        )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == (
        "value = 1\nvalue = 1\n"
    )


def test_apply_edit_relocates_anchor_with_extra_outer_newlines(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    apply_edit(
        context,
        FileEdit(path="app.py", old_text="\nvalue = 1\n\n", new_text="value = 2"),
    )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 2\n"


def test_apply_edit_uses_guarded_ast_aligned_line_range(tmp_path: Path):
    (tmp_path / "app.py").write_text(
        "def answer():\n    return 41\n", encoding="utf-8"
    )
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    apply_edit(
        context,
        FileEdit(
            path="app.py",
            old_text="    return 40",
            new_text="    return 42",
            start_line=2,
            end_line=2,
        ),
    )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == (
        "def answer():\n    return 42\n"
    )


def test_apply_edit_uses_line_range_to_disambiguate_repeated_anchor(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    apply_edit(
        context,
        FileEdit(
            path="app.py",
            old_text="value = 1",
            new_text="value = 2",
            start_line=2,
            end_line=2,
        ),
    )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == (
        "value = 1\nvalue = 2\n"
    )


def test_apply_edit_rejects_line_range_with_unrelated_guard(tmp_path: Path):
    original = "def answer():\n    return 41\n"
    (tmp_path / "app.py").write_text(original, encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    with pytest.raises(ToolExecutionError, match="does not resemble"):
        apply_edit(
            context,
            FileEdit(
                path="app.py",
                old_text="    raise RuntimeError('unrelated')",
                new_text="    return 42",
                start_line=2,
                end_line=2,
            ),
        )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original


def test_apply_edit_rejects_non_ast_aligned_python_range(tmp_path: Path):
    original = "def answer():\n    value = 41\n    return value\n"
    (tmp_path / "app.py").write_text(original, encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    with pytest.raises(ToolExecutionError, match="not aligned"):
        apply_edit(
            context,
            FileEdit(
                path="app.py",
                old_text="def answer():\n    value = 40",
                new_text="def answer():\n    value = 42",
                start_line=1,
                end_line=2,
            ),
        )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original


def test_apply_edit_rejects_line_range_that_breaks_python_syntax(tmp_path: Path):
    original = "def answer():\n    return 41\n"
    (tmp_path / "app.py").write_text(original, encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    with pytest.raises(ToolExecutionError, match="invalid Python"):
        apply_edit(
            context,
            FileEdit(
                path="app.py",
                old_text="    return 40",
                new_text="    return (",
                start_line=2,
                end_line=2,
            ),
        )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original


def test_file_edit_requires_complete_ordered_line_range():
    with pytest.raises(ValueError, match="provided together"):
        FileEdit(
            path="app.py",
            old_text="value = 1",
            new_text="value = 2",
            start_line=1,
        )
    with pytest.raises(ValueError, match="greater than or equal"):
        FileEdit(
            path="app.py",
            old_text="value = 1",
            new_text="value = 2",
            start_line=2,
            end_line=1,
        )


def test_apply_edits_preflights_complete_batch_before_writing(tmp_path: Path):
    (tmp_path / "first.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("value = 2\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    with pytest.raises(ToolExecutionError, match="exactly one"):
        apply_edits(
            context,
            [
                FileEdit(path="first.py", old_text="value = 1", new_text="value = 10"),
                FileEdit(path="second.py", old_text="missing", new_text="value = 20"),
            ],
        )

    assert (tmp_path / "first.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (tmp_path / "second.py").read_text(encoding="utf-8") == "value = 2\n"


def test_apply_edits_supports_sequential_edits_to_same_file(tmp_path: Path):
    (tmp_path / "app.py").write_text("first = 1\nsecond = 2\n", encoding="utf-8")
    context = ToolContext(Repository(tmp_path), {"write"}, [])

    paths = apply_edits(
        context,
        [
            FileEdit(path="app.py", old_text="first = 1", new_text="first = 10"),
            FileEdit(path="app.py", old_text="second = 2", new_text="second = 20"),
        ],
    )

    assert paths == ["app.py"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == (
        "first = 10\nsecond = 20\n"
    )


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
