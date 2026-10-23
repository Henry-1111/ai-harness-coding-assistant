from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from .knowledge import (
    KnowledgeIdentity,
    KnowledgeSource,
    KnowledgeTool,
    KnowledgeToolResult,
    RestrictedKnowledgeRegistry,
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

    def __init__(
        self,
        servers: Dict[str, str],
        timeout_seconds: float = 20.0,
        server_headers: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.servers = {name: url.rstrip("/") for name, url in servers.items()}
        self.timeout_seconds = timeout_seconds
        self.server_headers = server_headers or {}
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
                **self.server_headers.get(server, {}),
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


class MCPServerConfig(BaseModel):
    url: str = Field(pattern=r"^https?://")
    allowed_tenants: frozenset[str] = Field(min_length=1)
    allowed_projects: frozenset[str] = Field(default_factory=frozenset)
    headers_from_env: Dict[str, str] = Field(default_factory=dict)

    @field_validator("headers_from_env")
    @classmethod
    def validate_header_environment_names(cls, values: Dict[str, str]) -> Dict[str, str]:
        if any(
            not header.strip() or not environment.strip()
            for header, environment in values.items()
        ):
            raise ValueError("MCP header and environment variable names must be non-empty")
        return values


class MCPToolConfig(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    server: str = Field(min_length=1, max_length=64)
    remote_tool_name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=3, max_length=500)
    input_schema: Dict[str, Any]
    read_only: Literal[True] = True
    allowed_tenants: frozenset[str] = Field(min_length=1)
    allowed_projects: frozenset[str] = Field(default_factory=frozenset)


class MCPToolCatalogConfig(BaseModel):
    servers: Dict[str, MCPServerConfig]
    tools: List[MCPToolConfig]
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)

    @model_validator(mode="after")
    def validate_catalog(self) -> "MCPToolCatalogConfig":
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("MCP tool names must be unique")
        missing = sorted({tool.server for tool in self.tools} - set(self.servers))
        if missing:
            raise ValueError(f"MCP tools reference unknown servers: {missing}")
        for tool in self.tools:
            server = self.servers[tool.server]
            if not tool.allowed_tenants.issubset(server.allowed_tenants):
                raise ValueError(
                    f"tool {tool.name} grants tenants not allowed by server {tool.server}"
                )
        return self


class MCPToolCatalog:
    """Locally pinned MCP endpoints, schemas and tenant/project authorization policy."""

    def __init__(self, config: MCPToolCatalogConfig) -> None:
        self.config = config
        headers: Dict[str, Dict[str, str]] = {}
        for name, server in config.servers.items():
            resolved: Dict[str, str] = {}
            for header, environment_name in server.headers_from_env.items():
                value = os.getenv(environment_name)
                if not value:
                    raise ValueError(
                        "required MCP credential environment variable is missing: "
                        f"{environment_name}"
                    )
                resolved[header] = value
            headers[name] = resolved
        self.client = MCPClient(
            {name: server.url for name, server in config.servers.items()},
            timeout_seconds=config.timeout_seconds,
            server_headers=headers,
        )

    @classmethod
    def load(cls, path: Path) -> "MCPToolCatalog":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(MCPToolCatalogConfig.model_validate(raw))

    def register_authorized_tools(
        self,
        registry: RestrictedKnowledgeRegistry,
        *,
        tenant_id: str,
        project_id: str,
    ) -> tuple[frozenset[str], frozenset[str]]:
        allowed_tools = set()
        allowed_servers = set()
        for tool in self.config.tools:
            server = self.config.servers[tool.server]
            if not self._authorized(
                server.allowed_tenants,
                server.allowed_projects,
                tenant_id,
                project_id,
            ):
                continue
            if not self._authorized(
                tool.allowed_tenants,
                tool.allowed_projects,
                tenant_id,
                project_id,
            ):
                continue
            registry.register(
                MCPReadOnlyTool(
                    self.client,
                    server=tool.server,
                    remote_tool_name=tool.remote_tool_name,
                    name=tool.name,
                    description=tool.description,
                    input_schema=self._compile_schema(tool.name, tool.input_schema),
                ).as_knowledge_tool()
            )
            allowed_tools.add(tool.name)
            allowed_servers.add(tool.server)
        return frozenset(allowed_tools), frozenset(allowed_servers)

    @staticmethod
    def _authorized(
        tenants: frozenset[str],
        projects: frozenset[str],
        tenant_id: str,
        project_id: str,
    ) -> bool:
        return tenant_id in tenants and (not projects or project_id in projects)

    @staticmethod
    def _compile_schema(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        if schema.get("type") != "object":
            raise ValueError(f"MCP tool {name} input_schema must have type=object")
        if schema.get("additionalProperties", False):
            raise ValueError(f"MCP tool {name} cannot allow additional properties")
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not isinstance(properties, dict) or not required.issubset(properties):
            raise ValueError(f"MCP tool {name} has an invalid object schema")
        fields: Dict[str, tuple[Any, Any]] = {}
        for field_name, definition in properties.items():
            if not isinstance(definition, dict):
                raise ValueError(f"MCP tool {name} field {field_name} must be an object")
            annotation, constraints = MCPToolCatalog._compile_field(
                name, field_name, definition
            )
            default = ... if field_name in required else definition.get("default", None)
            if field_name not in required:
                annotation = Optional[annotation]
            fields[field_name] = (annotation, Field(default, **constraints))
        model_factory = cast(Any, create_model)
        return model_factory(
            f"MCP_{name}_Input",
            __config__=ConfigDict(extra="forbid", strict=True),
            **fields,
        )

    @staticmethod
    def _compile_field(
        tool_name: str, field_name: str, definition: Dict[str, Any]
    ) -> tuple[Any, Dict[str, Any]]:
        field_type = definition.get("type")
        scalar_types: Dict[str, Type[Any]] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }
        constraints: Dict[str, Any] = {}
        if field_type == "array":
            items = definition.get("items", {})
            item_type_name = items.get("type") if isinstance(items, dict) else None
            item_type = (
                scalar_types.get(item_type_name)
                if isinstance(item_type_name, str)
                else None
            )
            if item_type is None:
                raise ValueError(
                    f"MCP tool {tool_name} field {field_name} has unsupported array items"
                )
            annotation: Any = List[item_type]
            if "minItems" in definition:
                constraints["min_length"] = definition["minItems"]
            if "maxItems" in definition:
                constraints["max_length"] = definition["maxItems"]
            return annotation, constraints
        annotation = scalar_types.get(field_type) if isinstance(field_type, str) else None
        if annotation is None:
            raise ValueError(
                f"MCP tool {tool_name} field {field_name} has unsupported type: {field_type}"
            )
        constraint_names = {
            "minLength": "min_length",
            "maxLength": "max_length",
            "pattern": "pattern",
            "minimum": "ge",
            "maximum": "le",
        }
        for source, target in constraint_names.items():
            if source in definition:
                constraints[target] = definition[source]
        return annotation, constraints


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
                "arguments": arguments.model_dump(mode="json", exclude_none=True),
            },
            MCPIdentity(
                tenant_id=identity.tenant_id,
                project_id=identity.project_id,
                allowed_servers=identity.allowed_mcp_servers,
            ),
        )
        sources: List[KnowledgeSource] = []
        if isinstance(result, dict):
            if result.get("isError"):
                raise RuntimeError("MCP tool returned an error result")
            content = result.get("content", [])
            if isinstance(content, list):
                remaining_chars = 12_000
                for index, item in enumerate(content[:10]):
                    if not isinstance(item, dict) or item.get("type") != "text":
                        continue
                    text = item.get("text")
                    if not isinstance(text, str):
                        continue
                    text = text[:remaining_chars]
                    remaining_chars -= len(text)
                    if not text:
                        break
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
