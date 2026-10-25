from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set

from pydantic import BaseModel

from .models import FileEdit
from .repository import Repository


class ToolPermissionError(PermissionError):
    pass


class ToolExecutionError(RuntimeError):
    pass


class ToolSpec(BaseModel):
    name: str
    description: str
    required_permission: str


@dataclass
class ToolContext:
    repository: Repository
    permissions: Set[str]
    allowed_paths: List[str]

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


def apply_edit(context: ToolContext, edit: FileEdit) -> str:
    path = edit.path
    if not context.permits_path(path):
        raise ToolPermissionError(f"path is outside task scope: {path}")
    target = context.repository.resolve(path, must_exist=False)
    if target.exists():
        original = target.read_text(encoding="utf-8")
        if edit.old_text == "":
            raise ToolExecutionError("old_text must be non-empty for an existing file")
        anchor, occurrences = _unique_anchor(original, edit.old_text)
        if anchor is None:
            raise ToolExecutionError(
                f"expected exactly one old_text match in {path}, found {occurrences}"
            )
        replacement = edit.new_text
        if anchor != edit.old_text:
            numbered_replacement = _strip_numbered_snippet(replacement)
            if numbered_replacement is not None:
                replacement = numbered_replacement
        updated = original.replace(anchor, replacement, 1)
    else:
        if edit.old_text:
            raise ToolExecutionError("old_text must be empty when creating a file")
        target.parent.mkdir(parents=True, exist_ok=True)
        updated = edit.new_text
    target.write_text(updated, encoding="utf-8")
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
    return registry
