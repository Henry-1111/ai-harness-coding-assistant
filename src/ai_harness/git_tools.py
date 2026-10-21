from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class GitCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitChange:
    status: str
    path: str


class GitRepository:
    """Read-only Git context provider; all commands are fixed argument arrays."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        result = self._run("rev-parse", "--show-toplevel")
        if Path(result).resolve() != self.root:
            raise GitCommandError("root must be the Git repository top level")

    def head(self) -> str:
        return self._run("rev-parse", "HEAD")

    def branch(self) -> str:
        return self._run("branch", "--show-current")

    def changes(self) -> List[GitChange]:
        output = self._run("status", "--porcelain=v1", "-z")
        if not output:
            return []
        entries = output.split("\0")
        changes: List[GitChange] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            if not entry:
                break
            status = entry[:2]
            path = entry[3:]
            if status[0] in {"R", "C"} and index + 1 < len(entries):
                index += 1
                path = entries[index]
            changes.append(GitChange(status=status, path=path))
            index += 1
        return changes

    def diff(self, path: Optional[str] = None, staged: bool = False) -> str:
        arguments = ["diff", "--no-ext-diff", "--no-color"]
        if staged:
            arguments.append("--cached")
        if path:
            arguments.extend(["--", path])
        return self._run(*arguments)

    def recent_history(self, limit: int = 20) -> List[str]:
        if not 1 <= limit <= 100:
            raise ValueError("history limit must be within 1..100")
        output = self._run("log", f"-{limit}", "--format=%H%x09%s")
        return output.splitlines() if output else []

    def _run(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode != 0:
            raise GitCommandError(completed.stderr.strip() or "git command failed")
        return completed.stdout.rstrip("\n\0")
