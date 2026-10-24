from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def parse_cors_origins(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_CORS_ORIGINS
    origins: list[str] = []
    for raw_origin in value.split(","):
        origin = raw_origin.strip().rstrip("/")
        if not origin:
            continue
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError(f"invalid CORS origin: {raw_origin.strip()}")
        normalized = f"{parsed.scheme}://{parsed.netloc}"
        if normalized not in origins:
            origins.append(normalized)
    return tuple(origins)


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
    quota_reconciliation_interval_seconds: float
    cors_allowed_origins: tuple[str, ...]
    workspace_snapshot_directory: str
    workspace_snapshot_max_bytes: int
    workspace_snapshot_s3_bucket: str
    workspace_snapshot_s3_prefix: str
    workspace_snapshot_s3_endpoint: str
    workspace_snapshot_s3_region: str
    workspace_snapshot_s3_access_key_id: str
    workspace_snapshot_s3_secret_access_key: str

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
            quota_reconciliation_interval_seconds=float(
                os.getenv("AI_HARNESS_QUOTA_RECONCILIATION_INTERVAL_SECONDS", "30")
            ),
            cors_allowed_origins=parse_cors_origins(
                os.getenv("AI_HARNESS_CORS_ALLOWED_ORIGINS")
            ),
            workspace_snapshot_directory=os.getenv(
                "AI_HARNESS_WORKSPACE_SNAPSHOT_DIR", ""
            ),
            workspace_snapshot_max_bytes=int(
                os.getenv("AI_HARNESS_WORKSPACE_SNAPSHOT_MAX_BYTES", "10737418240")
            ),
            workspace_snapshot_s3_bucket=os.getenv(
                "AI_HARNESS_WORKSPACE_SNAPSHOT_S3_BUCKET", ""
            ),
            workspace_snapshot_s3_prefix=os.getenv(
                "AI_HARNESS_WORKSPACE_SNAPSHOT_S3_PREFIX", "ai-harness/workspaces"
            ),
            workspace_snapshot_s3_endpoint=os.getenv(
                "AI_HARNESS_WORKSPACE_SNAPSHOT_S3_ENDPOINT", ""
            ),
            workspace_snapshot_s3_region=os.getenv(
                "AI_HARNESS_WORKSPACE_SNAPSHOT_S3_REGION", ""
            ),
            workspace_snapshot_s3_access_key_id=os.getenv(
                "AI_HARNESS_WORKSPACE_SNAPSHOT_S3_ACCESS_KEY_ID", ""
            ),
            workspace_snapshot_s3_secret_access_key=os.getenv(
                "AI_HARNESS_WORKSPACE_SNAPSHOT_S3_SECRET_ACCESS_KEY", ""
            ),
        )
