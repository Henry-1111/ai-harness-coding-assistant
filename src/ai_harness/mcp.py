from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import httpx
from pydantic import BaseModel

from .knowledge import (
    KnowledgeIdentity,
    KnowledgeSource,
    KnowledgeTool,
    KnowledgeToolResult,
)


class MCPPermissionError(PermissionError):
    pass


@dataclass(frozen=True)
class MCPIdentity:
    tenant_id: str
    project_id: str
    allowed_servers: frozenset[str]


class MCPClient:
    """Identity-aware JSON-RPC client for allow-listed MCP HTTP endpoints."""

    def __init__(self, servers: Dict[str, str], timeout_seconds: float = 20.0) -> None:
        self.servers = {name: url.rstrip("/") for name, url in servers.items()}
        self.timeout_seconds = timeout_seconds
        self._request_id = 0

    def list_tools(self, server: str, identity: MCPIdentity) -> List[Dict[str, Any]]:
        result = self.call(server, "tools/list", {}, identity)
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return tools if isinstance(tools, list) else []

    def call(
        self,
        server: str,
        method: str,
        params: Dict[str, Any],
        identity: MCPIdentity,
    ) -> Any:
        if server not in identity.allowed_servers:
            raise MCPPermissionError(f"MCP server is not allowed for this identity: {server}")
        if server not in self.servers:
            raise KeyError(f"unknown MCP server: {server}")
        self._request_id += 1
        response = httpx.post(
            self.servers[server],
            headers={
                "X-Tenant-ID": identity.tenant_id,
                "X-Project-ID": identity.project_id,
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("error"):
            raise RuntimeError(str(body["error"]))
        return body.get("result")


class MCPReadOnlyTool:
    """Expose one MCP tool through the restricted, audited knowledge registry."""

    def __init__(
        self,
        client: MCPClient,
        *,
        server: str,
        remote_tool_name: str,
        name: str,
        description: str,
        input_schema: Type[BaseModel],
    ) -> None:
        self.client = client
        self.server = server
        self.remote_tool_name = remote_tool_name
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def as_knowledge_tool(self) -> KnowledgeTool:
        return KnowledgeTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            handler=self._call,
            read_only=True,
            backend="mcp",
        )

    def _call(self, arguments: BaseModel, identity: KnowledgeIdentity) -> KnowledgeToolResult:
        if self.server not in identity.allowed_mcp_servers:
            raise MCPPermissionError(f"MCP server is not allowed for this identity: {self.server}")
        result = self.client.call(
            self.server,
            "tools/call",
            {
                "name": self.remote_tool_name,
                "arguments": arguments.model_dump(mode="json"),
            },
            MCPIdentity(
                tenant_id=identity.tenant_id,
                project_id=identity.project_id,
                allowed_servers=identity.allowed_mcp_servers,
            ),
        )
        sources: List[KnowledgeSource] = []
        if isinstance(result, dict):
            content = result.get("content", [])
            if isinstance(content, list):
                for index, item in enumerate(content):
                    if not isinstance(item, dict) or item.get("type") != "text":
                        continue
                    text = item.get("text")
                    if not isinstance(text, str):
                        continue
                    sources.append(
                        KnowledgeSource(
                            source_id=f"mcp:{self.server}:{self.remote_tool_name}:{index}",
                            content=text,
                            path=item.get("path") if isinstance(item.get("path"), str) else None,
                            metadata={
                                "server": self.server,
                                "remote_tool": self.remote_tool_name,
                            },
                        )
                    )
        return KnowledgeToolResult(sources=sources)
