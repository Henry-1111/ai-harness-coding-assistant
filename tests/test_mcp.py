import pytest
from pydantic import BaseModel

from ai_harness.knowledge import KnowledgeIdentity, RestrictedKnowledgeRegistry
from ai_harness.mcp import MCPClient, MCPIdentity, MCPPermissionError, MCPReadOnlyTool


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
