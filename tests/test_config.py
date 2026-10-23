from ai_harness.config import Settings


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

    settings = Settings.from_environment()

    assert settings.job_workers == 4
    assert settings.job_poll_interval_seconds == 0.5
    assert settings.job_lease_seconds == 120
    assert settings.job_queue_capacity == 250
    assert settings.job_timeout_seconds == 900
    assert settings.circuit_failure_threshold == 5
    assert settings.circuit_recovery_seconds == 45
