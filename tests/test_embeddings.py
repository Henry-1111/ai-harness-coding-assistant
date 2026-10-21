from ai_harness.embeddings import OpenAICompatibleEmbeddingProvider


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }


def test_openai_compatible_embedding_provider_preserves_input_order(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("ai_harness.embeddings.httpx.post", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(
        "http://gateway.local/v1/", "tenant-key", "text-embedding"
    )

    vectors = provider.embed(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "http://gateway.local/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer tenant-key"
    assert captured["json"] == {"model": "text-embedding", "input": ["first", "second"]}
