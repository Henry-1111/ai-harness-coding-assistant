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
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from .embeddings import EmbeddingProvider
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


class DefinitionProvider(Protocol):
    def related_paths(self, query: str, hits: Sequence[SearchHit]) -> List[str]:
        ...


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
        embedding_provider: Optional[EmbeddingProvider] = None,
        definition_provider: Optional[DefinitionProvider] = None,
    ) -> None:
        self.repository = repository
        self.repo_map = repo_map
        self.changed_paths = list(dict.fromkeys(changed_paths or []))
        self.cache = cache or RetrievalCache()
        self.embedding_provider = embedding_provider
        self.definition_provider = definition_provider
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
                + (
                    type(self.embedding_provider).__name__ if self.embedding_provider else "",
                    type(self.definition_provider).__name__ if self.definition_provider else "",
                )
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
        windows: List[Tuple[str, int, List[str], str]] = []
        for path, text in texts.items():
            lines = text.splitlines()
            for start in range(0, max(len(lines), 1), 8):
                window_lines = lines[start : start + 12]
                windows.append((path, start, lines, "\n".join(window_lines)))
        document_frequency = {
            term: sum(term in window.lower() for _, _, _, window in windows) for term in terms
        }
        average_length = sum(len(window.split()) for _, _, _, window in windows) / max(
            len(windows), 1
        )
        lexical_scores: Dict[str, float] = {}
        representatives: Dict[str, SearchHit] = {}
        k1 = 1.5
        b = 0.75
        for path, start, lines, window_text in windows:
            lowered = window_text.lower()
            document_length = max(len(window_text.split()), 1)
            score = 0.0
            for term in terms:
                frequency = lowered.count(term)
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1.0
                    + (len(windows) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * document_length / max(average_length, 1.0)
                )
                score += inverse_frequency * frequency * (k1 + 1.0) / denominator
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
                    source="bm25_window",
                )
        lexical_rank = [
            path
            for path, _ in sorted(
                lexical_scores.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ]
        focus_ms = round((time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        semantic_scores: Dict[str, float] = {}
        semantic_error = ""
        semantic_candidates = list(
            dict.fromkeys(lexical_rank[:30] + structure_rank[:30] + forge_candidates[:30])
        )[:50]
        if self.embedding_provider and semantic_candidates:
            try:
                embedding_inputs = [
                    query,
                    *[
                        f"{path}\n{texts.get(path, '')[:2500]}"
                        for path in semantic_candidates
                    ],
                ]
                vectors = self.embedding_provider.embed(embedding_inputs)
                if len(vectors) != len(embedding_inputs):
                    raise ValueError("embedding result count mismatch")
                semantic_scores = {
                    path: self._cosine_similarity(vectors[0], vector)
                    for path, vector in zip(semantic_candidates, vectors[1:])
                }
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
        semantic_rank = [
            path
            for path, score in sorted(
                semantic_scores.items(), key=lambda pair: (-pair[1], pair[0])
            )
            if score > 0
        ]
        semantic_ms = round((time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        lsp_rank: List[str] = []
        lsp_error = ""
        if self.definition_provider:
            preliminary = [
                representatives.get(path) or self._representative(path, texts.get(path, ""))
                for path in (lexical_rank or structure_rank)[:5]
            ]
            try:
                lsp_rank = [
                    path
                    for path in self.definition_provider.related_paths(query, preliminary)
                    if path in texts
                ]
            except Exception as exc:
                lsp_error = f"{type(exc).__name__}: {exc}"
        lsp_ms = round((time.monotonic() - stage_started) * 1000)

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
            "semantic": (semantic_rank, 1.1),
            "deepen": (deepen_rank, 0.35),
            "lsp": (lsp_rank, 1.25),
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
                    details={"algorithm": "BM25", "k1": k1, "b": b},
                ),
                RetrievalStageTrace(
                    stage="semantic",
                    candidate_count=len(semantic_rank),
                    selected_paths=semantic_rank[:20],
                    duration_ms=semantic_ms,
                    details={
                        "enabled": self.embedding_provider is not None,
                        "error": semantic_error,
                    },
                ),
                RetrievalStageTrace(
                    stage="deepen",
                    candidate_count=len(deepen_rank),
                    selected_paths=deepen_rank[:20],
                    duration_ms=deepen_ms,
                ),
                RetrievalStageTrace(
                    stage="lsp",
                    candidate_count=len(lsp_rank),
                    selected_paths=lsp_rank[:20],
                    duration_ms=lsp_ms,
                    details={
                        "enabled": self.definition_provider is not None,
                        "error": lsp_error,
                    },
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

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)


class EvidenceJudge:
    """Deterministic sufficiency gate for bounded multi-round retrieval."""

    def __init__(self, minimum_coverage: float = 0.65) -> None:
        self.minimum_coverage = minimum_coverage

    def assess(self, query: str, hits: Sequence[SearchHit]) -> Tuple[bool, float, List[str]]:
        terms = ContextIndex._query_terms(query)
        searchable = " ".join(f"{hit.path} {hit.text}".lower() for hit in hits)
        missing = sorted(term for term in terms if term not in searchable)
        coverage = 1.0 - len(missing) / max(len(terms), 1)
        return bool(hits) and coverage >= self.minimum_coverage, coverage, missing


class ProgressiveRetriever:
    """Retrieve, judge evidence, refine the query and retrieve again within a hard budget."""

    def __init__(
        self,
        repository: Repository,
        repo_map: List[RepoFile],
        *,
        changed_paths: Optional[Sequence[str]] = None,
        cache: Optional[RetrievalCache] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        definition_provider: Optional[DefinitionProvider] = None,
        judge: Optional[EvidenceJudge] = None,
        max_rounds: int = 3,
    ) -> None:
        self.repository = repository
        self.repo_map = repo_map
        self.changed_paths = list(changed_paths or [])
        self.cache = cache or RetrievalCache()
        self.embedding_provider = embedding_provider
        self.definition_provider = definition_provider
        self.judge = judge or EvidenceJudge()
        self.max_rounds = max(1, max_rounds)
        self.last_trace = RetrievalTrace()

    def search(
        self,
        query: str,
        limit: int = 20,
        *,
        max_context_chars: int = 12_000,
    ) -> List[SearchHit]:
        selected: Dict[str, SearchHit] = {}
        stages: List[RetrievalStageTrace] = []
        queries: List[str] = []
        current_query = query
        stop_reason = "max_rounds"
        repository_chars = 0
        coverage = 0.0
        cache_hit = False

        for round_number in range(1, self.max_rounds + 1):
            queries.append(current_query)
            index = ContextIndex(
                self.repository,
                self.repo_map,
                changed_paths=self.changed_paths,
                cache=self.cache,
                embedding_provider=self.embedding_provider,
                definition_provider=self.definition_provider,
            )
            round_hits = index.search(
                current_query,
                limit=max(limit, 10),
                max_context_chars=max_context_chars,
            )
            cache_hit = cache_hit or index.last_trace.cache_hit
            repository_chars = max(repository_chars, index.last_trace.repository_chars)
            for stage in index.last_trace.stages:
                stage.round = round_number
                stages.append(stage)

            novel = 0
            for hit in round_hits:
                existing = selected.get(hit.path)
                if existing is None:
                    selected[hit.path] = hit
                    novel += 1
                elif hit.score > existing.score:
                    selected[hit.path] = hit

            # ContextIndex has already applied RRF and priors. Preserve that order;
            # sorting its display score again would apply lexical weight twice.
            ranked = list(selected.values())
            budgeted = self._apply_budget(ranked, limit, max_context_chars)
            sufficient, coverage, missing = self.judge.assess(query, budgeted)
            stages.append(
                RetrievalStageTrace(
                    stage="judge",
                    round=round_number,
                    candidate_count=len(budgeted),
                    selected_paths=[hit.path for hit in budgeted],
                    details={
                        "coverage": round(coverage, 6),
                        "missing_terms": missing,
                        "novel_paths": novel,
                        "sufficient": sufficient,
                    },
                )
            )
            if sufficient:
                stop_reason = "evidence_sufficient"
                break
            if sum(len(hit.text) for hit in budgeted) >= max_context_chars:
                stop_reason = "budget_exhausted"
                break
            if round_number >= self.max_rounds:
                stop_reason = "max_rounds"
                break
            if novel == 0:
                stop_reason = "no_new_evidence"
                break

            refined = self._refine_query(current_query, budgeted)
            if refined == current_query:
                stop_reason = "no_new_evidence"
                break
            stages.append(
                RetrievalStageTrace(
                    stage="refine",
                    round=round_number,
                    details={"previous_query": current_query, "refined_query": refined},
                )
            )
            current_query = refined

        final_hits = self._apply_budget(
            list(selected.values()),
            limit,
            max_context_chars,
        )
        context_chars = sum(len(hit.text) for hit in final_hits)
        self.last_trace = RetrievalTrace(
            queries=queries,
            stages=stages,
            cache_hit=cache_hit,
            query_coverage=round(coverage, 6),
            context_chars=context_chars,
            repository_chars=repository_chars,
            context_reduction_ratio=round(
                1.0 - min(context_chars / max(repository_chars, 1), 1.0), 6
            ),
            stop_reason=stop_reason,
            rounds=len(queries),
        )
        return final_hits

    def _refine_query(self, current_query: str, hits: Sequence[SearchHit]) -> str:
        hit_paths = {hit.path for hit in hits[:5]}
        related: List[str] = []
        for item in self.repo_map:
            if item.path not in hit_paths:
                continue
            related.extend(symbol.name for symbol in item.symbols[:5])
            related.extend(Path(name).name for name in item.imports[:5])
        current_terms = ContextIndex._query_terms(current_query)
        additions = [
            item
            for item in dict.fromkeys(related)
            if item.lower() not in current_terms
        ][:12]
        if not additions:
            return current_query
        return f"{current_query} {' '.join(additions)}"

    @staticmethod
    def _apply_budget(
        hits: Sequence[SearchHit], limit: int, max_context_chars: int
    ) -> List[SearchHit]:
        selected: List[SearchHit] = []
        used = 0
        for original in hits:
            if len(selected) >= limit or used >= max_context_chars:
                break
            hit = original.model_copy(deep=True)
            hit.text = hit.text[: max_context_chars - used]
            selected.append(hit)
            used += len(hit.text)
        return selected
