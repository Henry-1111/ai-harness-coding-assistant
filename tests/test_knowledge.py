from pathlib import Path

import pytest
from conftest import ScriptedModel
from pydantic import BaseModel, Field

from ai_harness.knowledge import (
    KnowledgeIdentity,
    KnowledgePermissionError,
    KnowledgeSource,
    KnowledgeTool,
    KnowledgeToolResult,
    RestrictedKnowledgeAgent,
    RestrictedKnowledgeRegistry,
    repository_knowledge_registry,
)
from ai_harness.repository import RepoMapBuilder, Repository


class QueryInput(BaseModel):
    query: str = Field(min_length=2)


def identity(**updates) -> KnowledgeIdentity:
    values = {
        "tenant_id": "tenant-a",
        "project_id": "project-a",
        "allowed_tools": frozenset({"search"}),
        "allowed_paths": ["src"],
    }
    values.update(updates)
    return KnowledgeIdentity(**values)


def test_restricted_react_stops_at_three_tool_calls():
    registry = RestrictedKnowledgeRegistry()
    registry.register(
        KnowledgeTool(
            name="search",
            description="search",
            input_schema=QueryInput,
            handler=lambda arguments, actor: KnowledgeToolResult(),
        )
    )
    model = ScriptedModel([{"action": "search", "arguments": {"query": "ab"}} for _ in range(4)])

    result = RestrictedKnowledgeAgent(model, registry).run("find evidence", identity())

    assert result.tool_calls == 3
    assert result.stop_reason == "max_tool_calls"
    assert len(model.calls) == 3


def test_registry_enforces_schema_authorization_and_source_scope():
    registry = RestrictedKnowledgeRegistry()
    registry.register(
        KnowledgeTool(
            name="search",
            description="search",
            input_schema=QueryInput,
            handler=lambda arguments, actor: KnowledgeToolResult(
                sources=[KnowledgeSource(source_id="secret", path="secret.txt", content="x")]
            ),
        )
    )

    with pytest.raises(KnowledgePermissionError, match="outside the authorized scope"):
        registry.execute("search", {"query": "ab"}, identity())
    with pytest.raises(KnowledgePermissionError, match="not allowed"):
        registry.execute(
            "search",
            {"query": "ab"},
            identity(allowed_tools=frozenset()),
        )
    with pytest.raises(ValueError):
        registry.execute("search", {"query": "x"}, identity())


@pytest.mark.parametrize("read_only,backend", [(False, "repository"), (True, "sql")])
def test_registry_rejects_write_and_direct_data_tools(read_only: bool, backend: str):
    registry = RestrictedKnowledgeRegistry()
    with pytest.raises(ValueError):
        registry.register(
            KnowledgeTool(
                name="unsafe",
                description="unsafe",
                input_schema=QueryInput,
                handler=lambda arguments, actor: KnowledgeToolResult(),
                read_only=read_only,
                backend=backend,
            )
        )


def test_repository_detail_tool_is_bounded_to_task_scope(sample_repository: Path):
    repository = Repository(sample_repository)
    registry = repository_knowledge_registry(repository, RepoMapBuilder().build(repository))

    result = registry.execute(
        "get_source_detail",
        {"path": "calculator.py", "start_line": 1, "end_line": 2},
        identity(
            allowed_tools=frozenset({"get_source_detail"}),
            allowed_paths=["calculator.py"],
        ),
    )

    assert result.sufficient is False
    assert "def add" in result.sources[0].content
    with pytest.raises(KnowledgePermissionError):
        registry.execute(
            "get_source_detail",
            {"path": "test_calculator.py"},
            identity(
                allowed_tools=frozenset({"get_source_detail"}),
                allowed_paths=["calculator.py"],
            ),
        )
