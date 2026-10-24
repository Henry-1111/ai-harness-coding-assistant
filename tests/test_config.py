import pytest

from ai_harness.config import Settings, parse_cors_origins


def test_embedding_endpoint_defaults_to_model_gateway(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_MODEL_BASE_URL", "http://gateway/v1")
    monkeypatch.setenv("AI_HARNESS_MODEL_API_KEY", "tenant-key")
    monkeypatch.delenv("AI_HARNESS_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("AI_HARNESS_EMBEDDING_API_KEY", raising=False)

    settings = Settings.from_environment()

    assert settings.embedding_base_url == "http://gateway/v1"
    assert settings.embedding_api_key == "tenant-key"


def test_embedding_endpoint_can_use_separate_gateway_credentials(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_MODEL_BASE_URL", "http://chat-gateway/v1")
    monkeypatch.setenv("AI_HARNESS_MODEL_API_KEY", "chat-key")
    monkeypatch.setenv("AI_HARNESS_EMBEDDING_BASE_URL", "http://embedding-gateway/v1")
    monkeypatch.setenv("AI_HARNESS_EMBEDDING_API_KEY", "embedding-key")

    settings = Settings.from_environment()

    assert settings.embedding_base_url == "http://embedding-gateway/v1"
    assert settings.embedding_api_key == "embedding-key"


def test_mcp_catalog_and_tenant_are_loaded_from_trusted_environment(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_MCP_CATALOG_PATH", "/etc/ai-harness/mcp-tools.json")
    monkeypatch.setenv("AI_HARNESS_TENANT_ID", "tenant-a")

    settings = Settings.from_environment()

    assert settings.mcp_catalog_path == "/etc/ai-harness/mcp-tools.json"
    assert settings.tenant_id == "tenant-a"


def test_job_worker_settings_are_configurable(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_JOB_WORKERS", "4")
    monkeypatch.setenv("AI_HARNESS_JOB_POLL_INTERVAL_SECONDS", "0.5")
    monkeypatch.setenv("AI_HARNESS_JOB_LEASE_SECONDS", "120")
    monkeypatch.setenv("AI_HARNESS_JOB_QUEUE_CAPACITY", "250")
    monkeypatch.setenv("AI_HARNESS_JOB_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("AI_HARNESS_CIRCUIT_FAILURE_THRESHOLD", "5")
    monkeypatch.setenv("AI_HARNESS_CIRCUIT_RECOVERY_SECONDS", "45")
    monkeypatch.setenv("AI_HARNESS_QUOTA_RECONCILIATION_INTERVAL_SECONDS", "15")

    settings = Settings.from_environment()

    assert settings.job_workers == 4
    assert settings.job_poll_interval_seconds == 0.5
    assert settings.job_lease_seconds == 120
    assert settings.job_queue_capacity == 250
    assert settings.job_timeout_seconds == 900
    assert settings.circuit_failure_threshold == 5
    assert settings.circuit_recovery_seconds == 45
    assert settings.quota_reconciliation_interval_seconds == 15


def test_request_authentication_settings_are_configurable(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_AUTH_ENABLED", "true")
    monkeypatch.setenv("AI_HARNESS_AUTH_CONFIG_PATH", "/run/secrets/governance.json")

    settings = Settings.from_environment()

    assert settings.auth_enabled is True
    assert settings.auth_config_path == "/run/secrets/governance.json"


def test_distributed_governance_settings_are_configurable(monkeypatch):
    monkeypatch.setenv(
        "AI_HARNESS_GOVERNANCE_DATABASE_URL",
        "postgresql://harness:secret@postgres/ai_harness",
    )
    monkeypatch.setenv("AI_HARNESS_REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("AI_HARNESS_REDIS_KEY_PREFIX", "test-harness")
    monkeypatch.setenv("AI_HARNESS_REDIS_QUOTA_TTL_SECONDS", "86400")
    monkeypatch.setenv("AI_HARNESS_REDIS_BINDING_CACHE_SECONDS", "60")

    settings = Settings.from_environment()

    assert settings.governance_database_url.endswith("/ai_harness")
    assert settings.redis_url == "redis://redis:6379/0"
    assert settings.redis_key_prefix == "test-harness"
    assert settings.redis_quota_ttl_seconds == 86400
    assert settings.redis_binding_cache_seconds == 60


def test_cors_origins_are_explicit_normalized_and_deduplicated(monkeypatch):
    monkeypatch.setenv(
        "AI_HARNESS_CORS_ALLOWED_ORIGINS",
        "https://console.example.com/, http://localhost:5173,https://console.example.com",
    )

    settings = Settings.from_environment()

    assert settings.cors_allowed_origins == (
        "https://console.example.com",
        "http://localhost:5173",
    )


@pytest.mark.parametrize(
    "value",
    ["*", "console.example.com", "https://user@example.com", "https://example.com/path"],
)
def test_cors_origins_reject_unsafe_or_malformed_values(value: str) -> None:
    with pytest.raises(ValueError, match="invalid CORS origin"):
        parse_cors_origins(value)
