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
