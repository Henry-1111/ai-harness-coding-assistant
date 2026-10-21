from __future__ import annotations

import ast
import fnmatch
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from .models import RepoFile, RepoSymbol, SearchHit

DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".txt",
    ".sql",
    ".html",
    ".css",
    ".scss",
}


class RepositoryBoundaryError(ValueError):
    pass


class Repository:
    def __init__(self, root: Path, excludes: Optional[Iterable[str]] = None) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise RepositoryBoundaryError(f"repository does not exist: {self.root}")
        self.excludes = set(excludes or DEFAULT_EXCLUDES)

    def resolve(self, relative_path: str, *, must_exist: bool = True) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise RepositoryBoundaryError("path escapes repository root") from exc
        if must_exist and not candidate.exists():
            raise FileNotFoundError(relative_path)
        return candidate

    def iter_files(self, scopes: Optional[Sequence[str]] = None) -> Iterable[Path]:
        patterns = list(scopes or [])
        for directory, names, filenames in os.walk(self.root):
            names[:] = sorted(name for name in names if name not in self.excludes)
            for filename in sorted(filenames):
                path = Path(directory) / filename
                relative = path.relative_to(self.root).as_posix()
                if path.suffix.lower() not in TEXT_EXTENSIONS:
                    continue
                if patterns and not any(
                    relative == pattern
                    or relative.startswith(pattern.rstrip("/") + "/")
                    or fnmatch.fnmatch(relative, pattern)
                    for pattern in patterns
                ):
                    continue
                yield path

    def read_text(self, relative_path: str) -> str:
        return self.resolve(relative_path).read_text(encoding="utf-8", errors="replace")


class RepoMapBuilder:
    def __init__(self, max_file_bytes: int = 512_000) -> None:
        self.max_file_bytes = max_file_bytes

    def build(
        self, repository: Repository, scopes: Optional[Sequence[str]] = None
    ) -> List[RepoFile]:
        result: List[RepoFile] = []
        for path in repository.iter_files(scopes):
            size = path.stat().st_size
            relative = path.relative_to(repository.root).as_posix()
            symbols: List[RepoSymbol] = []
            imports: List[str] = []
            if path.suffix == ".py" and size <= self.max_file_bytes:
                symbols, imports = self._python_structure(path)
            result.append(
                RepoFile(
                    path=relative,
                    language=self._language(path),
                    size=size,
                    symbols=symbols,
                    imports=imports,
                )
            )
        return result

    @staticmethod
    def _python_structure(path: Path) -> tuple[List[RepoSymbol], List[str]]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return [], []
        symbols: List[RepoSymbol] = []
        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(RepoSymbol(name=node.name, kind="class", line=node.lineno))
            elif isinstance(node, ast.AsyncFunctionDef):
                symbols.append(RepoSymbol(name=node.name, kind="async_function", line=node.lineno))
            elif isinstance(node, ast.FunctionDef):
                symbols.append(RepoSymbol(name=node.name, kind="function", line=node.lineno))
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        return sorted(symbols, key=lambda item: item.line), sorted(imports)

    @staticmethod
    def _language(path: Path) -> str:
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
        }.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "text")


class ContextIndex:
    """Progressive Forge -> Focus -> Deepen repository retrieval."""

    TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")

    def __init__(self, repository: Repository, repo_map: List[RepoFile]) -> None:
        self.repository = repository
        self.repo_map = repo_map

    def search(self, query: str, limit: int = 20) -> List[SearchHit]:
        terms = self._query_terms(query)
        if not terms:
            return []
        ranked_files = sorted(
            self.repo_map,
            key=lambda item: self._file_score(item, terms),
            reverse=True,
        )
        candidates = [item for item in ranked_files if self._file_score(item, terms) > 0]
        if not candidates:
            candidates = ranked_files[: min(20, len(ranked_files))]
        hits: List[SearchHit] = []
        for item in candidates[:50]:
            try:
                lines = self.repository.read_text(item.path).splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, text in enumerate(lines, start=1):
                lowered = text.lower()
                matched = sum(1 for term in terms if term in lowered)
                if matched:
                    start = max(0, line_number - 3)
                    end = min(len(lines), line_number + 2)
                    window = "\n".join(
                        f"{index + 1}: {lines[index]}" for index in range(start, end)
                    )
                    hits.append(
                        SearchHit(
                            path=item.path,
                            line=start + 1,
                            text=window[:2000],
                            score=float(matched) + self._file_score(item, terms),
                            source="text_window",
                        )
                    )
        return sorted(hits, key=lambda hit: (-hit.score, hit.path, hit.line))[:limit]

    @classmethod
    def _query_terms(cls, query: str) -> Set[str]:
        terms: Set[str] = set()
        for raw_term in cls.TOKEN_PATTERN.findall(query):
            term = raw_term.lower()
            terms.add(term)
            if all("\u4e00" <= character <= "\u9fff" for character in term):
                terms.update(term[index : index + 2] for index in range(len(term) - 1))
        return terms

    @staticmethod
    def _file_score(item: RepoFile, terms: Set[str]) -> float:
        path = item.path.lower()
        symbols = " ".join(symbol.name.lower() for symbol in item.symbols)
        imports = " ".join(item.imports).lower()
        return sum(
            3.0 if term in path else 2.0 if term in symbols else 0.5 if term in imports else 0.0
            for term in terms
        )
