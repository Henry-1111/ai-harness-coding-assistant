from __future__ import annotations

import ast
import fnmatch
import hashlib
import math
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import RepoFile, RepoSymbol, RetrievalStageTrace, RetrievalTrace, SearchHit

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

    def fingerprint(self, scopes: Optional[Sequence[str]] = None) -> str:
        """Stable cache key that changes whenever an indexed file changes."""
        digest = hashlib.sha256()
        for path in self.iter_files(scopes):
            stat = path.stat()
            relative = path.relative_to(self.root).as_posix()
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()


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


class QueryRewriter:
    """Deterministic query normalization used before repository retrieval."""

    ALIASES = {
        "bug": ("fix", "error", "failure", "exception"),
        "fix": ("bug", "repair", "error"),
        "timeout": ("deadline", "timed_out", "超时"),
        "quota": ("limit", "tokens", "配额"),
        "auth": ("authentication", "authorization", "api_key"),
        "retrieval": ("search", "context", "index", "召回"),
        "检索": ("召回", "搜索", "上下文", "索引"),
        "扣款": ("payment", "charge", "idempotency", "幂等"),
        "限流": ("rate_limit", "rpm", "tpm", "quota"),
        "恢复": ("resume", "checkpoint", "replay"),
    }

    CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

    def rewrite(self, query: str) -> List[str]:
        normalized = self.CAMEL_BOUNDARY.sub(" ", query).replace("_", " ").replace("-", " ")
        variants = [query.strip(), normalized.strip()]
        lowered = normalized.lower()
        expansions: List[str] = []
        for term, aliases in self.ALIASES.items():
            if term in lowered:
                expansions.extend(aliases)
        if expansions:
            variants.append(f"{normalized} {' '.join(expansions)}")
        return list(dict.fromkeys(item for item in variants if item))


class RetrievalCache:
    """Small bounded cache; repository fingerprints prevent stale retrieval results."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._values: "OrderedDict[str, Tuple[List[SearchHit], RetrievalTrace]]" = OrderedDict()

    def get(self, key: str) -> Optional[Tuple[List[SearchHit], RetrievalTrace]]:
        value = self._values.get(key)
        if value is None:
            return None
        self._values.move_to_end(key)
        hits, trace = value
        return [item.model_copy(deep=True) for item in hits], trace.model_copy(deep=True)

    def put(self, key: str, hits: List[SearchHit], trace: RetrievalTrace) -> None:
        self._values[key] = (
            [item.model_copy(deep=True) for item in hits],
            trace.model_copy(deep=True),
        )
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)


class ContextIndex:
    """Rewrite -> Forge -> Focus -> Deepen -> RRF selection with a context budget."""

    TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")

    def __init__(
        self,
        repository: Repository,
        repo_map: List[RepoFile],
        *,
        changed_paths: Optional[Sequence[str]] = None,
        cache: Optional[RetrievalCache] = None,
    ) -> None:
        self.repository = repository
        self.repo_map = repo_map
        self.changed_paths = list(dict.fromkeys(changed_paths or []))
        self.cache = cache or RetrievalCache()
        self.rewriter = QueryRewriter()
        self.last_trace = RetrievalTrace()

    def search(
        self,
        query: str,
        limit: int = 20,
        *,
        max_context_chars: int = 12_000,
    ) -> List[SearchHit]:
        started = time.monotonic()
        queries = self.rewriter.rewrite(query)
        terms = set().union(*(self._query_terms(item) for item in queries))
        if not terms:
            self.last_trace = RetrievalTrace(queries=queries, stop_reason="empty_query")
            return []

        fingerprint = self.repository.fingerprint()
        cache_key = hashlib.sha256(
            repr(
                (fingerprint, queries, limit, max_context_chars, sorted(self.changed_paths))
            ).encode()
        ).hexdigest()
        cached = self.cache.get(cache_key)
        if cached:
            hits, trace = cached
            trace.cache_hit = True
            trace.stages.append(
                RetrievalStageTrace(stage="select", details={"cache": "hit"})
            )
            self.last_trace = trace
            return hits

        rewrite_ms = round((time.monotonic() - started) * 1000)
        stage_started = time.monotonic()
        texts: Dict[str, str] = {}
        repository_chars = 0
        for item in self.repo_map:
            try:
                text = self.repository.read_text(item.path)
            except (OSError, UnicodeError):
                continue
            texts[item.path] = text
            repository_chars += len(text)

        structure_scores = {
            item.path: self._file_score(item, terms) for item in self.repo_map
        }
        structure_rank = [
            path
            for path, score in sorted(
                structure_scores.items(), key=lambda pair: (-pair[1], pair[0])
            )
            if score > 0
        ]
        forge_candidates = structure_rank or sorted(texts)[: min(50, len(texts))]
        forge_ms = round((time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        document_frequency = {
            term: sum(term in text.lower() for text in texts.values()) for term in terms
        }
        lexical_scores: Dict[str, float] = {}
        representatives: Dict[str, SearchHit] = {}
        for path, text in texts.items():
            lines = text.splitlines()
            for start in range(0, max(len(lines), 1), 8):
                window_lines = lines[start : start + 12]
                window_text = "\n".join(window_lines)
                lowered = window_text.lower()
                score = 0.0
                for term in terms:
                    frequency = lowered.count(term)
                    if not frequency:
                        continue
                    inverse_frequency = math.log(
                        (len(texts) + 1) / (document_frequency[term] + 1)
                    ) + 1.0
                    score += (1.0 + math.log(frequency)) * inverse_frequency
                if score <= 0:
                    continue
                if score > lexical_scores.get(path, 0.0):
                    lexical_scores[path] = score
                    numbered = "\n".join(
                        f"{line_number + 1}: {lines[line_number]}"
                        for line_number in range(start, min(start + 12, len(lines)))
                    )
                    representatives[path] = SearchHit(
                        path=path,
                        line=start + 1,
                        text=numbered[:3000],
                        score=score,
                        source="lexical_window",
                    )
        lexical_rank = [
            path
            for path, _ in sorted(
                lexical_scores.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ]
        focus_ms = round((time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        seed_paths = set((lexical_rank or structure_rank)[:10])
        seed_modules = {Path(path).stem for path in seed_paths}
        dependency_rank = [
            item.path
            for item in self.repo_map
            if item.path not in seed_paths
            and (set(item.imports) & seed_modules or Path(item.path).stem in seed_modules)
        ]
        git_rank = [path for path in self.changed_paths if path in texts]
        deepen_rank = list(dict.fromkeys(git_rank + sorted(dependency_rank)))
        deepen_ms = round((time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        rankings = {
            "lexical": (lexical_rank, 1.0),
            "structure": (structure_rank, 1.2),
            "deepen": (deepen_rank, 0.35),
        }
        fused: Dict[str, float] = {}
        sources: Dict[str, List[str]] = {}
        for source, (ranking, weight) in rankings.items():
            for rank, path in enumerate(ranking, start=1):
                fused[path] = fused.get(path, 0.0) + weight / (60 + rank)
                sources.setdefault(path, []).append(source)
        query_mentions_tests = any(
            term in {"test", "tests", "pytest", "unittest"} for term in terms
        )
        for path in fused:
            fused[path] += min(lexical_scores.get(path, 0.0), 20.0) * 0.001
            fused[path] += min(structure_scores.get(path, 0.0), 10.0) * 0.003
            if path.startswith("src/"):
                fused[path] += 0.012
            if path.startswith("tests/") and not query_mentions_tests:
                fused[path] -= 0.012
            if path.startswith("benchmarks/") or path.endswith("package-lock.json"):
                fused[path] -= 0.025
        if not fused:
            for rank, path in enumerate(forge_candidates, start=1):
                fused[path] = 1.0 / (60 + rank)
                sources[path] = ["forge"]

        selected: List[SearchHit] = []
        context_chars = 0
        for path, rrf_score in sorted(fused.items(), key=lambda pair: (-pair[1], pair[0])):
            if len(selected) >= limit or context_chars >= max_context_chars:
                break
            hit = representatives.get(path) or self._representative(path, texts.get(path, ""))
            remaining = max_context_chars - context_chars
            if remaining <= 0:
                break
            hit.text = hit.text[:remaining]
            hit.score = round(rrf_score * 1000 + hit.score, 6)
            hit.source = "hybrid_rrf[" + ",".join(sources[path]) + "]"
            selected.append(hit)
            context_chars += len(hit.text)

        searchable = " ".join(
            f"{hit.path} {hit.text}".lower() for hit in selected
        )
        covered = sum(1 for term in terms if term in searchable)
        coverage = covered / len(terms) if terms else 0.0
        if len(selected) >= limit or context_chars >= max_context_chars:
            stop_reason = "budget_exhausted"
        elif coverage >= 0.6:
            stop_reason = "evidence_sufficient"
        else:
            stop_reason = "candidates_exhausted"
        select_ms = round((time.monotonic() - stage_started) * 1000)
        reduction = 1.0 - min(context_chars / max(repository_chars, 1), 1.0)
        self.last_trace = RetrievalTrace(
            queries=queries,
            stages=[
                RetrievalStageTrace(
                    stage="rewrite",
                    duration_ms=rewrite_ms,
                    details={"terms": sorted(terms)},
                ),
                RetrievalStageTrace(
                    stage="forge",
                    candidate_count=len(forge_candidates),
                    selected_paths=forge_candidates[:20],
                    duration_ms=forge_ms,
                ),
                RetrievalStageTrace(
                    stage="focus",
                    candidate_count=len(lexical_rank),
                    selected_paths=lexical_rank[:20],
                    duration_ms=focus_ms,
                ),
                RetrievalStageTrace(
                    stage="deepen",
                    candidate_count=len(deepen_rank),
                    selected_paths=deepen_rank[:20],
                    duration_ms=deepen_ms,
                ),
                RetrievalStageTrace(
                    stage="select",
                    candidate_count=len(fused),
                    selected_paths=[hit.path for hit in selected],
                    duration_ms=select_ms,
                    details={"limit": limit, "max_context_chars": max_context_chars},
                ),
            ],
            query_coverage=round(coverage, 6),
            context_chars=context_chars,
            repository_chars=repository_chars,
            context_reduction_ratio=round(reduction, 6),
            stop_reason=stop_reason,
        )
        self.cache.put(cache_key, selected, self.last_trace)
        return selected

    @staticmethod
    def _representative(path: str, text: str) -> SearchHit:
        lines = text.splitlines()
        numbered = "\n".join(
            f"{index + 1}: {line}" for index, line in enumerate(lines[:20])
        )
        return SearchHit(path=path, line=1, text=numbered[:3000], source="structure")

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
