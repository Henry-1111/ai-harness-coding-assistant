from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Protocol, Type

from pydantic import BaseModel, Field, ValidationError

from .agents import AgentModel
from .models import RepoFile, SearchHit
from .repository import ContextIndex, Repository


class KnowledgePermissionError(PermissionError):
    pass


class KnowledgeToolExecutionError(RuntimeError):
    pass


class KnowledgeSource(BaseModel):
    source_id: str
    content: str
    path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeToolResult(BaseModel):
    sources: List[KnowledgeSource] = Field(default_factory=list)
    sufficient: bool = False


class KnowledgeObservation(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    sources: List[KnowledgeSource] = Field(default_factory=list)
    error: Optional[str] = None


class KnowledgeDecision(BaseModel):
    action: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    answer: str = ""


class KnowledgeRunResult(BaseModel):
    sources: List[KnowledgeSource] = Field(default_factory=list)
    observations: List[KnowledgeObservation] = Field(default_factory=list)
    tool_calls: int = 0
    stop_reason: str


class KnowledgeIdentity(BaseModel):
    tenant_id: str
    project_id: str
    allowed_tools: frozenset[str]
    allowed_paths: List[str] = Field(default_factory=list)
    allowed_mcp_servers: frozenset[str] = Field(default_factory=frozenset)

    def permits_path(self, path: str) -> bool:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return False
        if not self.allowed_paths:
            return True
        return any(
            path == scope
            or path.startswith(scope.rstrip("/") + "/")
            or fnmatch.fnmatch(path, scope)
            for scope in self.allowed_paths
        )


class KnowledgeToolHandler(Protocol):
    def __call__(
        self, arguments: BaseModel, identity: KnowledgeIdentity
    ) -> KnowledgeToolResult: ...


@dataclass(frozen=True)
class KnowledgeTool:
    name: str
    description: str
    input_schema: Type[BaseModel]
    handler: KnowledgeToolHandler
    read_only: bool = True
    backend: str = "repository"


class RestrictedKnowledgeRegistry:
    """Allow-listed, schema-validated registry for read-only knowledge tools."""

    FORBIDDEN_BACKENDS = {"sql", "redis", "filesystem-write", "shell"}

    def __init__(self) -> None:
        self._tools: Dict[str, KnowledgeTool] = {}

    def register(self, tool: KnowledgeTool) -> None:
        if not tool.read_only:
            raise ValueError("restricted ReAct tools must be read-only")
        if tool.backend.lower() in self.FORBIDDEN_BACKENDS:
            raise ValueError(f"backend is forbidden in restricted ReAct: {tool.backend}")
        self._tools[tool.name] = tool

    def descriptions(self, identity: KnowledgeIdentity) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.model_json_schema(),
            }
            for name, tool in sorted(self._tools.items())
            if name in identity.allowed_tools
        ]

    def execute(
        self, name: str, raw_arguments: Dict[str, Any], identity: KnowledgeIdentity
    ) -> KnowledgeToolResult:
        if name not in identity.allowed_tools:
            raise KnowledgePermissionError(f"tool is not allowed for this identity: {name}")
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown knowledge tool: {name}")
        arguments = tool.input_schema.model_validate(raw_arguments)
        try:
            result = KnowledgeToolResult.model_validate(tool.handler(arguments, identity))
        except (KeyError, PermissionError, ValidationError, ValueError):
            raise
        except Exception as exc:
            raise KnowledgeToolExecutionError(
                f"knowledge tool failed safely: {type(exc).__name__}: {exc}"
            ) from exc
        forbidden = [
            source.path
            for source in result.sources
            if source.path and not identity.permits_path(source.path)
        ]
        if forbidden:
            raise KnowledgePermissionError(
                f"tool returned sources outside the authorized scope: {forbidden}"
            )
        return result


class RestrictedKnowledgeAgent:
    SYSTEM = """You are a bounded knowledge scout. Choose only from the supplied read-only tools.
Return JSON only: {"action": tool name or "finish", "arguments": object, "answer": string}.
Use finish when evidence is sufficient or no useful read-only action remains. Never request writes,
shell commands, SQL, Redis, credentials, or paths outside the authorized scope."""

    def __init__(
        self,
        model: AgentModel,
        registry: RestrictedKnowledgeRegistry,
        *,
        max_tool_calls: int = 3,
    ) -> None:
        if not 1 <= max_tool_calls <= 3:
            raise ValueError("restricted ReAct max_tool_calls must be within 1..3")
        self.model = model
        self.registry = registry
        self.max_tool_calls = max_tool_calls

    def run(self, goal: str, identity: KnowledgeIdentity) -> KnowledgeRunResult:
        observations: List[KnowledgeObservation] = []
        sources: Dict[str, KnowledgeSource] = {}
        stop_reason = "max_tool_calls"
        for _ in range(self.max_tool_calls):
            try:
                decision = KnowledgeDecision.model_validate(
                    self.model.complete_json(
                        system=self.SYSTEM,
                        payload={
                            "goal": goal,
                            "tools": self.registry.descriptions(identity),
                            "observations": [item.model_dump(mode="json") for item in observations],
                            "remaining_tool_calls": self.max_tool_calls - len(observations),
                        },
                    )
                )
            except ValidationError:
                stop_reason = "invalid_decision"
                break
            if decision.action == "finish":
                stop_reason = "model_finished"
                break
            try:
                result = self.registry.execute(decision.action, decision.arguments, identity)
                observation = KnowledgeObservation(
                    tool=decision.action,
                    arguments=decision.arguments,
                    sources=result.sources,
                )
                for source in result.sources:
                    sources[source.source_id] = source
                observations.append(observation)
                if result.sufficient:
                    stop_reason = "evidence_sufficient"
                    break
            except (
                KeyError,
                PermissionError,
                KnowledgeToolExecutionError,
                ValidationError,
                ValueError,
            ) as exc:
                observations.append(
                    KnowledgeObservation(
                        tool=decision.action,
                        arguments=decision.arguments,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return KnowledgeRunResult(
            sources=list(sources.values()),
            observations=observations,
            tool_calls=len(observations),
            stop_reason=stop_reason,
        )


class SearchProjectKnowledgeInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


class GetSourceDetailInput(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=200, ge=1)


def repository_knowledge_registry(
    repository: Repository, repo_map: List[RepoFile]
) -> RestrictedKnowledgeRegistry:
    registry = RestrictedKnowledgeRegistry()

    def search(arguments: BaseModel, identity: KnowledgeIdentity) -> KnowledgeToolResult:
        parsed = SearchProjectKnowledgeInput.model_validate(arguments)
        hits = ContextIndex(repository, repo_map).search(parsed.query, limit=parsed.limit)
        return KnowledgeToolResult(
            sources=[
                KnowledgeSource(
                    source_id=f"search:{hit.path}:{hit.line}",
                    path=hit.path,
                    content=hit.text,
                    metadata={"line": hit.line, "score": hit.score, "source": hit.source},
                )
                for hit in hits
                if identity.permits_path(hit.path)
            ]
        )

    def detail(arguments: BaseModel, identity: KnowledgeIdentity) -> KnowledgeToolResult:
        parsed = GetSourceDetailInput.model_validate(arguments)
        if parsed.end_line < parsed.start_line or parsed.end_line - parsed.start_line > 400:
            raise ValueError("source detail range must be ordered and at most 400 lines")
        if not identity.permits_path(parsed.path):
            raise KnowledgePermissionError(f"path is outside the authorized scope: {parsed.path}")
        lines = repository.read_text(parsed.path).splitlines()
        content = "\n".join(lines[parsed.start_line - 1 : parsed.end_line])
        return KnowledgeToolResult(
            sources=[
                KnowledgeSource(
                    source_id=f"detail:{parsed.path}:{parsed.start_line}-{parsed.end_line}",
                    path=parsed.path,
                    content=content,
                    metadata={"start_line": parsed.start_line, "end_line": parsed.end_line},
                )
            ]
        )

    registry.register(
        KnowledgeTool(
            name="search_project_knowledge",
            description="Search repository code and documentation for additional evidence.",
            input_schema=SearchProjectKnowledgeInput,
            handler=search,
        )
    )
    registry.register(
        KnowledgeTool(
            name="get_source_detail",
            description="Read a bounded line range from one authorized repository file.",
            input_schema=GetSourceDetailInput,
            handler=detail,
        )
    )
    return registry


def sources_to_hits(sources: List[KnowledgeSource]) -> List[SearchHit]:
    return [
        SearchHit(
            path=source.path,
            line=int(source.metadata.get("line", source.metadata.get("start_line", 1))),
            text=source.content,
            score=float(source.metadata.get("score", 0.0)),
            source="restricted_react",
        )
        for source in sources
        if source.path
    ]
