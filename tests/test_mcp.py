import pytest

from ai_harness.mcp import MCPClient, MCPIdentity, MCPPermissionError


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
