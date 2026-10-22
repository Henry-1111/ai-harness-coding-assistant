import pytest
from pydantic import BaseModel

from ai_harness.knowledge import (
    KnowledgeIdentity,
    RestrictedKnowledgeRegistry,
    sources_to_hits,
)
from ai_harness.mcp import (
    MCPClient,
    MCPIdentity,
    MCPPermissionError,
    MCPReadOnlyTool,
    MCPToolCatalog,
    MCPToolCatalogConfig,
)


def test_mcp_identity_rejects_unapproved_server():
    client = MCPClient({"knowledge": "http://localhost:9000/mcp"})
    identity = MCPIdentity(
        tenant_id="tenant-a",
        project_id="project-a",
        allowed_servers=frozenset(),
    )

    with pytest.raises(MCPPermissionError, match="not allowed"):
        client.call("knowledge", "tools/list", {}, identity)


def test_mcp_unknown_server_is_distinct_from_permission_failure():
    client = MCPClient({})
    identity = MCPIdentity(
        tenant_id="tenant-a",
        project_id="project-a",
        allowed_servers=frozenset({"missing"}),
    )

    with pytest.raises(KeyError, match="unknown MCP server"):
        client.call("missing", "tools/list", {}, identity)


class SearchInput(BaseModel):
    query: str


def test_mcp_tool_adapter_preserves_identity_and_read_only_policy(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "approved context"}]},
            }

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("ai_harness.mcp.httpx.post", fake_post)
    registry = RestrictedKnowledgeRegistry()
    registry.register(
        MCPReadOnlyTool(
            MCPClient({"knowledge": "http://localhost:9000/mcp"}),
            server="knowledge",
            remote_tool_name="search",
            name="search_mcp",
            description="search knowledge",
            input_schema=SearchInput,
        ).as_knowledge_tool()
    )
    actor = KnowledgeIdentity(
        tenant_id="tenant-a",
        project_id="project-a",
        allowed_tools=frozenset({"search_mcp"}),
        allowed_mcp_servers=frozenset({"knowledge"}),
    )

    result = registry.execute("search_mcp", {"query": "quota"}, actor)

    assert result.sources[0].content == "approved context"
    assert captured["headers"]["X-Tenant-ID"] == "tenant-a"
    assert captured["json"]["method"] == "tools/call"

    actor.allowed_mcp_servers = frozenset()
    with pytest.raises(MCPPermissionError):
        registry.execute("search_mcp", {"query": "quota"}, actor)


def catalog_config(**tool_updates):
    tool = {
        "name": "search_docs",
        "server": "knowledge",
        "remote_tool_name": "search",
        "description": "search project documentation",
        "allowed_tenants": ["tenant-a"],
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 20},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["query"],
        },
    }
    tool.update(tool_updates)
    return MCPToolCatalogConfig.model_validate(
        {
            "servers": {
                "knowledge": {
                    "url": "http://localhost:9000/mcp",
                    "allowed_tenants": ["tenant-a"],
                    "headers_from_env": {"Authorization": "TEST_MCP_TOKEN"},
                }
            },
            "tools": [tool],
        }
    )


def test_catalog_registers_only_authorized_tools_and_validates_arguments(monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "Bearer secret")
    catalog = MCPToolCatalog(catalog_config())
    registry = RestrictedKnowledgeRegistry()

    tools, servers = catalog.register_authorized_tools(
        registry, tenant_id="tenant-a", project_id="project-a"
    )

    assert tools == frozenset({"search_docs"})
    assert servers == frozenset({"knowledge"})
    actor = KnowledgeIdentity(
        tenant_id="tenant-a",
        project_id="project-a",
        allowed_tools=tools,
        allowed_mcp_servers=servers,
    )
    with pytest.raises(ValueError):
        registry.execute("search_docs", {"query": "x"}, actor)
    with pytest.raises(ValueError):
        registry.execute("search_docs", {"query": "quota", "unexpected": True}, actor)
    with pytest.raises(ValueError):
        registry.execute("search_docs", {"query": "quota", "limit": "3"}, actor)

    denied_registry = RestrictedKnowledgeRegistry()
    denied_tools, denied_servers = catalog.register_authorized_tools(
        denied_registry, tenant_id="tenant-b", project_id="project-a"
    )
    assert denied_tools == frozenset()
    assert denied_servers == frozenset()


def test_catalog_resolves_secret_headers_and_keeps_pathless_mcp_sources(monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "Bearer secret")
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "external evidence"}]},
            }

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("ai_harness.mcp.httpx.post", fake_post)
    catalog = MCPToolCatalog(catalog_config())
    registry = RestrictedKnowledgeRegistry()
    tools, servers = catalog.register_authorized_tools(
        registry, tenant_id="tenant-a", project_id="project-a"
    )
    actor = KnowledgeIdentity(
        tenant_id="tenant-a",
        project_id="project-a",
        allowed_tools=tools,
        allowed_mcp_servers=servers,
    )

    result = registry.execute("search_docs", {"query": "quota"}, actor)

    assert captured["headers"]["Authorization"] == "Bearer secret"
    hits = sources_to_hits(result.sources)
    assert hits[0].path.startswith("mcp:knowledge:search:")
    assert hits[0].text == "external evidence"


def test_catalog_rejects_unsafe_or_inconsistent_configuration(monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "Bearer secret")
    with pytest.raises(ValueError, match="grants tenants"):
        catalog_config(allowed_tenants=["tenant-b"])
    with pytest.raises(ValueError):
        catalog_config(read_only=False)
    with pytest.raises(ValueError, match="unsupported type"):
        MCPToolCatalog(
            catalog_config(
                input_schema={
                    "type": "object",
                    "properties": {"payload": {"type": "object"}},
                }
            )
        ).register_authorized_tools(
            RestrictedKnowledgeRegistry(),
            tenant_id="tenant-a",
            project_id="project-a",
        )
