import pytest

from ai_harness.cli import build_promotion_publisher
from ai_harness.config import Settings, parse_cors_origins, secret_from_environment
from ai_harness.promotion import GitHubPullRequestPublisher, GitLabMergeRequestPublisher


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


def test_secrets_can_be_loaded_from_files(monkeypatch, tmp_path) -> None:
    model_key = tmp_path / "model-key"
    model_key.write_text("model-secret\n")
    gitlab_token = tmp_path / "gitlab-token"
    gitlab_token.write_text("gitlab-secret\n")
    database_url = tmp_path / "database-url"
    database_url.write_text("postgresql://harness:secret@postgres/ai_harness\n")
    monkeypatch.delenv("AI_HARNESS_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("AI_HARNESS_PROMOTION_GITLAB_TOKEN", raising=False)
    monkeypatch.delenv("AI_HARNESS_GOVERNANCE_DATABASE_URL", raising=False)
    monkeypatch.setenv("AI_HARNESS_MODEL_API_KEY_FILE", str(model_key))
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_TOKEN_FILE", str(gitlab_token))
    monkeypatch.setenv("AI_HARNESS_GOVERNANCE_DATABASE_URL_FILE", str(database_url))

    settings = Settings.from_environment()

    assert settings.model_api_key == "model-secret"
    assert settings.embedding_api_key == "model-secret"
    assert settings.promotion_gitlab_token == "gitlab-secret"
    assert settings.governance_database_url.endswith("@postgres/ai_harness")


def test_secret_file_rejects_ambiguous_or_unreadable_configuration(
    monkeypatch, tmp_path
) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("from-file")
    monkeypatch.setenv("EXAMPLE_SECRET", "inline")
    monkeypatch.setenv("EXAMPLE_SECRET_FILE", str(secret_file))

    with pytest.raises(ValueError, match="configure only one"):
        secret_from_environment("EXAMPLE_SECRET")

    monkeypatch.delenv("EXAMPLE_SECRET")
    monkeypatch.setenv("EXAMPLE_SECRET_FILE", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="cannot read"):
        secret_from_environment("EXAMPLE_SECRET")


def test_secret_file_rejects_empty_and_oversized_values(monkeypatch, tmp_path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text(" \n")
    monkeypatch.setenv("EXAMPLE_SECRET_FILE", str(secret_file))
    with pytest.raises(ValueError, match="is empty"):
        secret_from_environment("EXAMPLE_SECRET")

    secret_file.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds"):
        secret_from_environment("EXAMPLE_SECRET")


def test_mcp_catalog_and_tenant_are_loaded_from_trusted_environment(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_MCP_CATALOG_PATH", "/etc/ai-harness/mcp-tools.json")
    monkeypatch.setenv("AI_HARNESS_TENANT_ID", "tenant-a")

    settings = Settings.from_environment()

    assert settings.mcp_catalog_path == "/etc/ai-harness/mcp-tools.json"
    assert settings.tenant_id == "tenant-a"


def test_multi_proposal_is_opt_in(monkeypatch):
    monkeypatch.delenv("AI_HARNESS_MULTI_PROPOSAL_ENABLED", raising=False)
    assert Settings.from_environment().multi_proposal_enabled is False

    monkeypatch.setenv("AI_HARNESS_MULTI_PROPOSAL_ENABLED", "true")
    assert Settings.from_environment().multi_proposal_enabled is True


def test_job_worker_settings_are_configurable(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_JOB_WORKERS", "4")
    monkeypatch.setenv("AI_HARNESS_JOB_POLL_INTERVAL_SECONDS", "0.5")
    monkeypatch.setenv("AI_HARNESS_JOB_LEASE_SECONDS", "120")
    monkeypatch.setenv("AI_HARNESS_JOB_QUEUE_CAPACITY", "250")
    monkeypatch.setenv("AI_HARNESS_JOB_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("AI_HARNESS_CIRCUIT_FAILURE_THRESHOLD", "5")
    monkeypatch.setenv("AI_HARNESS_CIRCUIT_RECOVERY_SECONDS", "45")
    monkeypatch.setenv("AI_HARNESS_QUOTA_RECONCILIATION_INTERVAL_SECONDS", "15")
    monkeypatch.setenv("AI_HARNESS_API_MAX_REQUEST_BODY_BYTES", "2097152")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_DIR", "/shared/snapshots")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_MAX_BYTES", "1048576")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_S3_BUCKET", "workspace-snapshots")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_S3_PREFIX", "production/harness")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_S3_REGION", "us-east-1")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_S3_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AI_HARNESS_WORKSPACE_SNAPSHOT_S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_MODE", "git_branch")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GIT_REMOTE_URL", "git@example.com:repo.git")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GIT_BASE_REF", "develop")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GIT_BRANCH_PREFIX", "review/harness")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_REPOSITORY", "acme/project")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_TOKEN", "github-token")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_API_URL", "https://github.example/api/v3")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_API_VERSION", "2022-11-28")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_RETRY_BACKOFF_SECONDS", "0.5")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_DRAFT", "false")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_API_URL", "https://gitlab.example/api/v4")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_RETRY_BACKOFF_SECONDS", "0.75")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_DRAFT", "false")

    settings = Settings.from_environment()

    assert settings.job_workers == 4
    assert settings.job_poll_interval_seconds == 0.5
    assert settings.job_lease_seconds == 120
    assert settings.job_queue_capacity == 250
    assert settings.job_timeout_seconds == 900
    assert settings.circuit_failure_threshold == 5
    assert settings.circuit_recovery_seconds == 45
    assert settings.quota_reconciliation_interval_seconds == 15
    assert settings.api_max_request_body_bytes == 2097152
    assert settings.workspace_snapshot_directory == "/shared/snapshots"
    assert settings.workspace_snapshot_max_bytes == 1048576
    assert settings.workspace_snapshot_s3_bucket == "workspace-snapshots"
    assert settings.workspace_snapshot_s3_prefix == "production/harness"
    assert settings.workspace_snapshot_s3_endpoint == "http://minio:9000"
    assert settings.workspace_snapshot_s3_region == "us-east-1"
    assert settings.workspace_snapshot_s3_access_key_id == "access"
    assert settings.workspace_snapshot_s3_secret_access_key == "secret"
    assert settings.promotion_mode == "git_branch"
    assert settings.promotion_git_remote_url == "git@example.com:repo.git"
    assert settings.promotion_git_base_ref == "develop"
    assert settings.promotion_git_branch_prefix == "review/harness"
    assert settings.promotion_github_repository == "acme/project"
    assert settings.promotion_github_token == "github-token"
    assert settings.promotion_github_api_url == "https://github.example/api/v3"
    assert settings.promotion_github_api_version == "2022-11-28"
    assert settings.promotion_github_timeout_seconds == 15
    assert settings.promotion_github_max_attempts == 4
    assert settings.promotion_github_retry_backoff_seconds == 0.5
    assert settings.promotion_github_draft is False
    assert settings.promotion_gitlab_api_url == "https://gitlab.example/api/v4"
    assert settings.promotion_gitlab_timeout_seconds == 20
    assert settings.promotion_gitlab_max_attempts == 5
    assert settings.promotion_gitlab_retry_backoff_seconds == 0.75
    assert settings.promotion_gitlab_draft is False


def test_request_authentication_settings_are_configurable(monkeypatch):
    monkeypatch.setenv("AI_HARNESS_AUTH_ENABLED", "true")
    monkeypatch.setenv("AI_HARNESS_AUTH_CONFIG_PATH", "/run/secrets/governance.json")

    settings = Settings.from_environment()

    assert settings.auth_enabled is True
    assert settings.auth_config_path == "/run/secrets/governance.json"


def test_github_pull_request_publisher_requires_complete_configuration(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("AI_HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AI_HARNESS_PROMOTION_MODE", "git_branch")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GIT_REMOTE_URL", "git@example.com:repo.git")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_REPOSITORY", "acme/project")
    monkeypatch.delenv("AI_HARNESS_PROMOTION_GITHUB_TOKEN", raising=False)

    with pytest.raises(SystemExit, match="must be configured together"):
        build_promotion_publisher(Settings.from_environment())

    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_TOKEN", "github-token")
    publisher = build_promotion_publisher(Settings.from_environment())

    assert isinstance(publisher, GitHubPullRequestPublisher)


def test_gitlab_merge_request_publisher_requires_complete_configuration(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("AI_HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AI_HARNESS_PROMOTION_MODE", "git_branch")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GIT_REMOTE_URL", "git@example.com:repo.git")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_PROJECT", "acme/project")
    monkeypatch.delenv("AI_HARNESS_PROMOTION_GITLAB_TOKEN", raising=False)

    with pytest.raises(SystemExit, match="must be configured together"):
        build_promotion_publisher(Settings.from_environment())

    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_TOKEN", "gitlab-token")
    publisher = build_promotion_publisher(Settings.from_environment())

    assert isinstance(publisher, GitLabMergeRequestPublisher)


def test_github_and_gitlab_promotion_are_mutually_exclusive(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI_HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AI_HARNESS_PROMOTION_MODE", "git_branch")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GIT_REMOTE_URL", "git@example.com:repo.git")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_REPOSITORY", "acme/project")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITHUB_TOKEN", "github-token")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_PROJECT", "acme/project")
    monkeypatch.setenv("AI_HARNESS_PROMOTION_GITLAB_TOKEN", "gitlab-token")

    with pytest.raises(SystemExit, match="only one"):
        build_promotion_publisher(Settings.from_environment())


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
