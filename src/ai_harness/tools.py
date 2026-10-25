from __future__ import annotations

import ast
import difflib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

from pydantic import BaseModel

from .models import FileEdit
from .repository import Repository


class ToolPermissionError(PermissionError):
    pass


class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, *, path: str = "", code: str = "tool_error") -> None:
        super().__init__(message)
        self.path = path
        self.code = code


class ToolSpec(BaseModel):
    name: str
    description: str
    required_permission: str


@dataclass
class ToolContext:
    repository: Repository
    permissions: Set[str]
    allowed_paths: List[str]
    audit_events: List[Dict[str, Any]] = field(default_factory=list)

    def permits_path(self, path: str) -> bool:
        if not self.allowed_paths:
            return True
        return any(
            path == scope or path.startswith(scope.rstrip("/") + "/")
            for scope in self.allowed_paths
        )


class ToolRegistry:
    """Validate model-requested operations before any side effect occurs."""

    def __init__(self) -> None:
        self._tools: Dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    def execute(self, name: str, context: ToolContext, **arguments: Any) -> Any:
        if name not in self._tools:
            raise ToolExecutionError(f"unknown tool: {name}")
        spec, handler = self._tools[name]
        if spec.required_permission not in context.permissions:
            raise ToolPermissionError(f"tool {name} requires permission {spec.required_permission}")
        path = arguments.get("path")
        if isinstance(path, str) and not context.permits_path(path):
            raise ToolPermissionError(f"path is outside task scope: {path}")
        return handler(context, **arguments)

    def specs(self) -> List[ToolSpec]:
        return [item[0] for item in self._tools.values()]


def read_file(context: ToolContext, path: str) -> str:
    return context.repository.read_text(path)


def _strip_numbered_snippet(text: str) -> str | None:
    """Remove retrieval display line numbers from a copied multi-line snippet."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return None
    numbers: List[int] = []
    stripped: List[str] = []
    for line in lines:
        ending = "\n" if line.endswith("\n") else ""
        content = line[:-1] if ending else line
        if content.endswith("\r"):
            content = content[:-1]
        match = re.fullmatch(r"\s*(\d+): ?(.*)", content)
        if match is None:
            return None
        numbers.append(int(match.group(1)))
        stripped.append(match.group(2) + ending)
    if any(right != left + 1 for left, right in zip(numbers, numbers[1:])):
        return None
    return "".join(stripped)


def _unique_anchor(original: str, old_text: str) -> tuple[str | None, int]:
    occurrences = original.count(old_text)
    if occurrences == 1:
        return old_text, occurrences
    if occurrences > 1:
        return None, occurrences

    candidates: List[str] = []
    numbered = _strip_numbered_snippet(old_text)
    if numbered is not None:
        candidates.append(numbered)
    normalized_newlines = old_text.replace("\r\n", "\n").replace("\r", "\n")
    candidates.extend(
        [
            normalized_newlines,
            normalized_newlines.strip("\n"),
            textwrap.dedent(normalized_newlines),
        ]
    )
    seen = {old_text}
    ambiguous = 0
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidate_occurrences = original.count(candidate)
        if candidate_occurrences == 1:
            return candidate, candidate_occurrences
        ambiguous = max(ambiguous, candidate_occurrences)
    return None, ambiguous


def _normalized_edit_text(text: str) -> str:
    numbered = _strip_numbered_snippet(text)
    if numbered is not None:
        text = numbered
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(
        line.rstrip() for line in textwrap.dedent(normalized).strip("\n").splitlines()
    )


def _python_range_is_ast_aligned(source: str, start_line: int, end_line: int) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.stmt)
            and getattr(node, "lineno", None) == start_line
            and getattr(node, "end_lineno", None) == end_line
        ):
            return True
        for _field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            statements = [item for item in value if isinstance(item, ast.stmt)]
            for first_index, first in enumerate(statements):
                if getattr(first, "lineno", None) != start_line:
                    continue
                for last in statements[first_index:]:
                    if getattr(last, "end_lineno", None) == end_line:
                        return True
    return False


def _replace_line_range(path: str, original: str, edit: FileEdit) -> str:
    start_line = edit.start_line
    end_line = edit.end_line
    if start_line is None or end_line is None:
        raise ToolExecutionError(
            f"old_text could not be safely located in {path}",
            path=path,
            code="anchor_not_unique",
        )
    lines = original.splitlines(keepends=True)
    if start_line > len(lines) or end_line > len(lines):
        raise ToolExecutionError(
            f"line range {start_line}-{end_line} is outside {path}",
            path=path,
            code="line_range_out_of_bounds",
        )
    selected = "".join(lines[start_line - 1 : end_line])
    expected = _normalized_edit_text(edit.old_text)
    actual = _normalized_edit_text(selected)
    similarity = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False).ratio()
    if not expected or similarity < 0.7:
        raise ToolExecutionError(
            f"line range {start_line}-{end_line} does not resemble old_text in {path}",
            path=path,
            code="line_range_guard_mismatch",
        )
    if path.endswith(".py") and not _python_range_is_ast_aligned(
        original, start_line, end_line
    ):
        raise ToolExecutionError(
            f"line range {start_line}-{end_line} is not aligned to Python statements in {path}",
            path=path,
            code="line_range_not_ast_aligned",
        )
    replacement = edit.new_text
    numbered_replacement = _strip_numbered_snippet(replacement)
    if numbered_replacement is not None:
        replacement = numbered_replacement
    if selected.endswith(("\n", "\r")) and replacement and not replacement.endswith("\n"):
        replacement += "\n"
    updated = "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])
    if path.endswith(".py"):
        try:
            ast.parse(updated)
        except SyntaxError as exc:
            raise ToolExecutionError(
                f"line-range edit would produce invalid Python in {path}: {exc.msg}",
                path=path,
                code="invalid_python_edit",
            ) from exc
    return updated


def _updated_text(path: str, original: str | None, edit: FileEdit) -> tuple[str, str]:
    if original is not None:
        if edit.old_text == "":
            raise ToolExecutionError(
                "old_text must be non-empty for an existing file",
                path=path,
                code="empty_existing_anchor",
            )
        anchor, occurrences = _unique_anchor(original, edit.old_text)
        if anchor is None:
            if edit.start_line is not None:
                return _replace_line_range(path, original, edit), "line_range"
            raise ToolExecutionError(
                f"expected exactly one old_text match in {path}, found {occurrences}",
                path=path,
                code="anchor_not_unique",
            )
        replacement = edit.new_text
        if anchor != edit.old_text:
            numbered_replacement = _strip_numbered_snippet(replacement)
            if numbered_replacement is not None:
                replacement = numbered_replacement
        strategy = "exact_anchor" if anchor == edit.old_text else "normalized_anchor"
        return original.replace(anchor, replacement, 1), strategy
    if edit.old_text:
        raise ToolExecutionError(
            "old_text must be empty when creating a file",
            path=path,
            code="new_file_has_anchor",
        )
    return edit.new_text, "new_file"


def apply_edits(context: ToolContext, edits: List[FileEdit]) -> List[str]:
    """Preflight every edit, then write the complete decision without partial patches."""
    staged: Dict[str, tuple[Path, str | None, str]] = {}
    staged_audit_events: List[Dict[str, str]] = []
    ordered_paths: List[str] = []
    for edit in edits:
        path = edit.path
        if not context.permits_path(path):
            raise ToolPermissionError(f"path is outside task scope: {path}")
        target = context.repository.resolve(path, must_exist=False)
        if path in staged:
            _, initial, current = staged[path]
        else:
            initial = target.read_text(encoding="utf-8") if target.exists() else None
            current = initial
            ordered_paths.append(path)
        updated, strategy = _updated_text(path, current, edit)
        staged[path] = (target, initial, updated)
        staged_audit_events.append(
            {"tool": "apply_edits", "path": path, "strategy": strategy}
        )

    written: List[str] = []
    try:
        for path in ordered_paths:
            target, _, updated = staged[path]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(updated, encoding="utf-8")
            written.append(path)
    except OSError:
        for path in reversed(written):
            target, initial, _ = staged[path]
            if initial is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(initial, encoding="utf-8")
        raise
    context.audit_events.extend(staged_audit_events)
    return ordered_paths


def apply_edit(context: ToolContext, edit: FileEdit) -> str:
    path = edit.path
    apply_edits(context, [edit])
    return path


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a scoped repository file",
            required_permission="read",
        ),
        read_file,
    )
    registry.register(
        ToolSpec(
            name="apply_edit",
            description="Apply one exact, scoped edit",
            required_permission="write",
        ),
        apply_edit,
    )
    registry.register(
        ToolSpec(
            name="apply_edits",
            description="Preflight and apply one complete scoped edit batch",
            required_permission="write",
        ),
        apply_edits,
    )
    return registry
