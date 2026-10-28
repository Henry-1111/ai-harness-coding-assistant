from pathlib import Path

import pytest

from ai_harness.repository import (
    ContextIndex,
    EvidenceJudge,
    ProgressiveRetriever,
    QueryRewriter,
    RepoMapBuilder,
    Repository,
    RepositoryBoundaryError,
    RetrievalCache,
)


class FakeEmbeddings:
    def embed(self, texts):
        return [
            [1.0, 0.0] if "authorization" in text else [0.0, 1.0]
            for text in texts
        ]


class FakeDefinitions:
    def related_paths(self, query, hits):
        return ["src/policy.py"]


def test_repo_map_extracts_python_symbols_and_imports(tmp_path: Path):
    (tmp_path / "service.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "class Gateway:\n"
        "    pass\n\n"
        "async def run_task():\n"
        "    return Path('.')\n",
        encoding="utf-8",
    )
    repo_map = RepoMapBuilder().build(Repository(tmp_path))

    assert len(repo_map) == 1
    assert [symbol.name for symbol in repo_map[0].symbols] == ["Gateway", "run_task"]
    assert repo_map[0].imports == ["json", "pathlib"]


def test_repo_map_extracts_javascript_declarations_and_static_imports(tmp_path: Path):
    (tmp_path / "gateway.js").write_text(
        '// function fake() {}\n'
        'const example = "class NotAClass {}";\n'
        'import { quota } from "./quota";\n'
        'export async function processTask() {}\n'
        'const retryRequest = async () => true;\n'
        'class Gateway { async send() {} }\n'
        'const Created = class {};\n'
        'const helper = require("../shared/format");\n'
        'export { quota } from "./quota-public";\n',
        encoding="utf-8",
    )

    item = RepoMapBuilder().build(Repository(tmp_path))[0]

    assert [(symbol.name, symbol.kind, symbol.line) for symbol in item.symbols] == [
        ("processTask", "async_function", 4),
        ("retryRequest", "async_function", 5),
        ("Gateway", "class", 6),
        ("Gateway.send", "method", 6),
        ("Created", "class", 7),
    ]
    assert item.imports == ["../shared/format", "./quota", "./quota-public"]


def test_repo_map_omits_dynamic_javascript_imports(tmp_path: Path):
    (tmp_path / "dynamic.js").write_text(
        'const lazy = import("./lazy");\n'
        "const unknown = require(moduleName);\n",
        encoding="utf-8",
    )

    item = RepoMapBuilder().build(Repository(tmp_path))[0]

    assert item.imports == []


@pytest.mark.parametrize(
    ("suffix", "source", "expected"),
    [
        (".mjs", "export function load() {}\n", [("load", "function")]),
        (".cjs", "const load = () => true;\n", [("load", "function")]),
        (
            ".jsx",
            "export default function Dashboard() { return <main />; }\n",
            [("Dashboard", "function")],
        ),
        (
            ".ts",
            "import type { Config } from './types';\n"
            "export interface Service { run(): void }\n"
            "export type Result = string;\n"
            "export const createService = (): Service => ({ run() {} });\n",
            [
                ("Service", "interface"),
                ("Service.run", "method"),
                ("Result", "type"),
                ("createService", "function"),
                ("run", "method"),
            ],
        ),
        (
            ".tsx",
            "export const Widget = () => <span />;\n",
            [("Widget", "function")],
        ),
        (".mts", "export function load(): void {}\n", [("load", "function")]),
        (".cts", "export function load(): void {}\n", [("load", "function")]),
    ],
)
def test_repo_map_extracts_jsx_and_typescript_symbols(
    tmp_path: Path, suffix: str, source: str, expected: list[tuple[str, str]]
):
    target = tmp_path / f"component{suffix}"
    target.write_text(source, encoding="utf-8")

    item = RepoMapBuilder().build(Repository(tmp_path))[0]

    assert [(symbol.name, symbol.kind) for symbol in item.symbols] == expected
    if suffix == ".ts":
        assert item.imports == ["./types"]


def test_repo_map_skips_oversized_javascript_file(tmp_path: Path):
    (tmp_path / "large.js").write_text("function shouldNotIndex() {}\n", encoding="utf-8")

    item = RepoMapBuilder(max_file_bytes=10).build(Repository(tmp_path))[0]

    assert item.symbols == []


def test_repo_map_extracts_go_types_methods_and_imports(tmp_path: Path):
    (tmp_path / "gateway.go").write_text(
        "package gateway\n"
        "// func fake() {}\n"
        'import ("context"; quota "example.com/project/quota")\n'
        "type Gateway struct{}\n"
        "type Runner interface { Run() error }\n"
        "type Alias = string\n"
        "func NewGateway() *Gateway { return nil }\n"
        "func (g *Gateway) Reserve() {}\n",
        encoding="utf-8",
    )

    item = RepoMapBuilder().build(Repository(tmp_path))[0]

    assert [(symbol.name, symbol.kind, symbol.line) for symbol in item.symbols] == [
        ("Gateway", "struct", 4),
        ("Runner", "interface", 5),
        ("Runner.Run", "method", 5),
        ("Alias", "type", 6),
        ("NewGateway", "function", 7),
        ("Gateway.Reserve", "method", 8),
    ]
    assert item.imports == ["context", "example.com/project/quota"]


def test_repo_map_extracts_java_types_methods_and_imports(tmp_path: Path):
    (tmp_path / "Gateway.java").write_text(
        "package com.example;\n"
        "// class Fake {}\n"
        "import java.util.List;\n"
        "import static com.example.Quota.reserve;\n"
        "import com.example.*;\n"
        "class Gateway { Gateway() {} void reserve() {} }\n"
        "interface Service { void run(); }\n"
        "record Item(String id) {}\n"
        "enum State { READY; }\n",
        encoding="utf-8",
    )

    item = RepoMapBuilder().build(Repository(tmp_path))[0]

    assert [(symbol.name, symbol.kind) for symbol in item.symbols] == [
        ("Gateway", "class"),
        ("Gateway.Gateway", "method"),
        ("Gateway.reserve", "method"),
        ("Service", "interface"),
        ("Service.run", "method"),
        ("Item", "record"),
        ("State", "enum"),
    ]
    assert item.imports == ["com.example.*", "com.example.Quota", "java.util.List"]


@pytest.mark.parametrize("suffix", [".go", ".java"])
def test_repo_map_skips_oversized_go_and_java_files(tmp_path: Path, suffix: str):
    (tmp_path / f"large{suffix}").write_text("class Ignore {}\n" * 10, encoding="utf-8")

    item = RepoMapBuilder(max_file_bytes=10).build(Repository(tmp_path))[0]

    assert item.symbols == []


def test_context_search_ranks_symbol_and_text_matches(tmp_path: Path):
    (tmp_path / "quota.py").write_text(
        "class QuotaManager:\n    def reserve_tokens(self):\n        return 'quota reservation'\n",
        encoding="utf-8",
    )
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("fix quota reservation")

    assert hits
    assert hits[0].path == "quota.py"
    assert any("reservation" in hit.text for hit in hits)
    assert hits[0].source.startswith("hybrid_rrf")


def test_context_search_adds_chinese_bigrams(tmp_path: Path):
    (tmp_path / "payment.py").write_text("# 防止重复扣款\n", encoding="utf-8")
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("修复订单重复扣款问题")

    assert hits and hits[0].path == "payment.py"


def test_repository_rejects_path_escape(tmp_path: Path):
    repository = Repository(tmp_path)
    with pytest.raises(RepositoryBoundaryError):
        repository.resolve("../outside.txt", must_exist=False)


def test_repository_honors_file_scope(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("x = 1", encoding="utf-8")

    files = list(Repository(tmp_path).iter_files(["src"]))

    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["src/app.py"]


def test_query_rewriter_expands_domain_terms():
    rewritten = QueryRewriter().rewrite("修复 token 限流 timeout")

    assert any("rpm" in item and "deadline" in item for item in rewritten)


def test_hybrid_retrieval_fuses_structure_lexical_and_git_context(tmp_path: Path):
    (tmp_path / "quota.py").write_text(
        "class QuotaManager:\n    def reserve_tokens(self):\n        return 'reserved'\n",
        encoding="utf-8",
    )
    (tmp_path / "worker.py").write_text("import quota\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("value = 1\n" * 200, encoding="utf-8")
    repository = Repository(tmp_path)
    index = ContextIndex(
        repository,
        RepoMapBuilder().build(repository),
        changed_paths=["quota.py"],
    )

    hits = index.search("reserve quota tokens", limit=2)

    assert hits[0].path == "quota.py"
    assert "lexical" in hits[0].source
    assert index.last_trace.stages[-1].selected_paths[0] == "quota.py"
    assert index.last_trace.context_reduction_ratio > 0


def test_typescript_symbol_retrieval_expands_relative_imports(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "quota.ts").write_text("export function reserveTokens() {}\n", encoding="utf-8")
    (source / "consumer.ts").write_text(
        'import * as quota from "./quota";\nexport const run = () => quota;\n',
        encoding="utf-8",
    )
    (source / "unrelated.ts").write_text("export const value = 1;\n", encoding="utf-8")
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("reserveTokens", limit=3)

    assert hits[0].path == "src/quota.ts"
    assert "structure" in hits[0].source
    assert "src/consumer.ts" in next(
        stage.selected_paths for stage in index.last_trace.stages if stage.stage == "deepen"
    )


@pytest.mark.parametrize(
    ("target_name", "target_source", "consumer_name", "consumer_source"),
    [
        (
            "quota.go",
            "package quota\nfunc ReserveTokens() {}\n",
            "handler.go",
            'package handler\nimport "example.com/acme/quota"\n',
        ),
        (
            "Quota.java",
            "package com.acme;\nclass Quota { void reserveTokens() {} }\n",
            "Handler.java",
            "package com.acme;\nimport com.acme.Quota;\nclass Handler {}\n",
        ),
    ],
)
def test_go_java_retrieval_expands_imported_modules(
    tmp_path: Path,
    target_name: str,
    target_source: str,
    consumer_name: str,
    consumer_source: str,
):
    source = tmp_path / "src"
    source.mkdir()
    (source / target_name).write_text(target_source, encoding="utf-8")
    (source / consumer_name).write_text(consumer_source, encoding="utf-8")
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("reserveTokens", limit=2)

    assert hits[0].path == f"src/{target_name}"
    assert f"src/{consumer_name}" in next(
        stage.selected_paths for stage in index.last_trace.stages if stage.stage == "deepen"
    )


def test_java_fixture_retrieves_symbol_and_dependent_file():
    fixture = Path(__file__).parent / "fixtures" / "multilang" / "java"
    repository = Repository(fixture)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("reserveTokens", limit=2)

    assert hits[0].path == "example/Quota.java"
    assert "example/Handler.java" in next(
        stage.selected_paths for stage in index.last_trace.stages if stage.stage == "deepen"
    )


def test_retrieval_cache_hits_and_repository_change_invalidates_it(tmp_path: Path):
    target = tmp_path / "payments.py"
    target.write_text("def reserve_payment():\n    return True\n", encoding="utf-8")
    repository = Repository(tmp_path)
    cache = RetrievalCache()

    first = ContextIndex(repository, RepoMapBuilder().build(repository), cache=cache)
    first.search("reserve payment")
    second = ContextIndex(repository, RepoMapBuilder().build(repository), cache=cache)
    second.search("reserve payment")
    assert second.last_trace.cache_hit is True

    target.write_text("def reserve_payment():\n    return False\n", encoding="utf-8")
    third = ContextIndex(repository, RepoMapBuilder().build(repository), cache=cache)
    third.search("reserve payment")
    assert third.last_trace.cache_hit is False


def test_context_budget_caps_selected_text(tmp_path: Path):
    (tmp_path / "large.py").write_text(
        "\n".join(f"quota token reservation line {index}" for index in range(100)),
        encoding="utf-8",
    )
    repository = Repository(tmp_path)
    index = ContextIndex(repository, RepoMapBuilder().build(repository))

    hits = index.search("quota token reservation", max_context_chars=80)

    assert sum(len(hit.text) for hit in hits) <= 80
    assert index.last_trace.stop_reason == "budget_exhausted"


def test_hybrid_retrieval_fuses_embeddings_and_lsp_definitions(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "handler.py").write_text(
        "def authorization_handler():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "src" / "policy.py").write_text(
        "def allow_request():\n    return True\n", encoding="utf-8"
    )
    repository = Repository(tmp_path)
    index = ContextIndex(
        repository,
        RepoMapBuilder().build(repository),
        embedding_provider=FakeEmbeddings(),
        definition_provider=FakeDefinitions(),
    )

    hits = index.search("authorization", limit=2)

    assert {hit.path for hit in hits} == {"src/handler.py", "src/policy.py"}
    assert any(
        stage.stage == "semantic" and stage.candidate_count
        for stage in index.last_trace.stages
    )
    assert any(stage.stage == "lsp" and stage.selected_paths for stage in index.last_trace.stages)


def test_progressive_retrieval_refines_until_evidence_is_sufficient(tmp_path: Path):
    (tmp_path / "quota.py").write_text(
        "class QuotaManager:\n    def reserve_tokens(self):\n        return True\n",
        encoding="utf-8",
    )
    repository = Repository(tmp_path)
    retriever = ProgressiveRetriever(
        repository,
        RepoMapBuilder().build(repository),
        judge=EvidenceJudge(minimum_coverage=1.0),
        max_rounds=2,
    )

    hits = retriever.search("quota reserve missingconcept", limit=3)

    assert hits[0].path == "quota.py"
    assert retriever.last_trace.rounds == 2
    assert any(stage.stage == "refine" for stage in retriever.last_trace.stages)
    assert retriever.last_trace.stop_reason == "max_rounds"


def test_progressive_retrieval_preserves_first_round_rrf_order(tmp_path: Path):
    (tmp_path / "primary.py").write_text(
        "def exact_quota_reservation():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "secondary.py").write_text(
        "# quota\n" * 20, encoding="utf-8"
    )
    repository = Repository(tmp_path)
    repo_map = RepoMapBuilder().build(repository)
    baseline = ContextIndex(repository, repo_map).search("exact quota reservation", limit=2)
    progressive = ProgressiveRetriever(repository, repo_map).search(
        "exact quota reservation", limit=2
    )

    assert [hit.path for hit in progressive] == [hit.path for hit in baseline]
