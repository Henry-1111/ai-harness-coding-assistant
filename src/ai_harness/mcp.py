from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import httpx


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
