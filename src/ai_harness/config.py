from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_directory: Path
    model_base_url: str
    model_api_key: str
    model_name: str
    model_timeout_seconds: float
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    lsp_command: str
    sandbox_mode: str
    sandbox_image: str
    restricted_react_enabled: bool
    restricted_react_max_tool_calls: int
    mcp_catalog_path: str
    tenant_id: str
    auth_enabled: bool
    auth_config_path: str
    governance_database_url: str
    redis_url: str
    redis_key_prefix: str
    redis_quota_ttl_seconds: int
    redis_binding_cache_seconds: int
    job_workers: int
    job_poll_interval_seconds: float
    job_lease_seconds: float
    job_queue_capacity: int
    job_timeout_seconds: float
    circuit_failure_threshold: int
    circuit_recovery_seconds: float

    @classmethod
    def from_environment(cls) -> "Settings":
        model_base_url = os.getenv("AI_HARNESS_MODEL_BASE_URL", "https://api.openai.com/v1")
        model_api_key = os.getenv("AI_HARNESS_MODEL_API_KEY", "")
        return cls(
            data_directory=Path(os.getenv("AI_HARNESS_DATA_DIR", ".ai-harness")),
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_name=os.getenv("AI_HARNESS_MODEL", "gpt-4.1-mini"),
            model_timeout_seconds=float(os.getenv("AI_HARNESS_MODEL_TIMEOUT", "60")),
            embedding_base_url=(os.getenv("AI_HARNESS_EMBEDDING_BASE_URL") or model_base_url),
            embedding_api_key=(os.getenv("AI_HARNESS_EMBEDDING_API_KEY") or model_api_key),
            embedding_model=os.getenv("AI_HARNESS_EMBEDDING_MODEL", ""),
            lsp_command=os.getenv("AI_HARNESS_LSP_COMMAND", ""),
            sandbox_mode=os.getenv("AI_HARNESS_SANDBOX_MODE", "docker"),
            sandbox_image=os.getenv("AI_HARNESS_SANDBOX_IMAGE", "python:3.12-slim"),
            restricted_react_enabled=(
                os.getenv("AI_HARNESS_RESTRICTED_REACT_ENABLED", "false").lower()
                in {"1", "true", "yes", "on"}
            ),
            restricted_react_max_tool_calls=int(
                os.getenv("AI_HARNESS_RESTRICTED_REACT_MAX_TOOL_CALLS", "3")
            ),
            mcp_catalog_path=os.getenv("AI_HARNESS_MCP_CATALOG_PATH", ""),
            tenant_id=os.getenv("AI_HARNESS_TENANT_ID", "local"),
            auth_enabled=(
                os.getenv("AI_HARNESS_AUTH_ENABLED", "false").lower()
                in {"1", "true", "yes", "on"}
            ),
            auth_config_path=os.getenv("AI_HARNESS_AUTH_CONFIG_PATH", ""),
            governance_database_url=os.getenv("AI_HARNESS_GOVERNANCE_DATABASE_URL", ""),
            redis_url=os.getenv("AI_HARNESS_REDIS_URL", ""),
            redis_key_prefix=os.getenv("AI_HARNESS_REDIS_KEY_PREFIX", "ai-harness"),
            redis_quota_ttl_seconds=int(
                os.getenv("AI_HARNESS_REDIS_QUOTA_TTL_SECONDS", "259200")
            ),
            redis_binding_cache_seconds=int(
                os.getenv("AI_HARNESS_REDIS_BINDING_CACHE_SECONDS", "300")
            ),
            job_workers=int(os.getenv("AI_HARNESS_JOB_WORKERS", "2")),
            job_poll_interval_seconds=float(
                os.getenv("AI_HARNESS_JOB_POLL_INTERVAL_SECONDS", "0.2")
            ),
            job_lease_seconds=float(os.getenv("AI_HARNESS_JOB_LEASE_SECONDS", "300")),
            job_queue_capacity=int(os.getenv("AI_HARNESS_JOB_QUEUE_CAPACITY", "100")),
            job_timeout_seconds=float(os.getenv("AI_HARNESS_JOB_TIMEOUT_SECONDS", "1800")),
            circuit_failure_threshold=int(
                os.getenv("AI_HARNESS_CIRCUIT_FAILURE_THRESHOLD", "3")
            ),
            circuit_recovery_seconds=float(
                os.getenv("AI_HARNESS_CIRCUIT_RECOVERY_SECONDS", "30")
            ),
        )
