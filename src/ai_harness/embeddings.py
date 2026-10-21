from __future__ import annotations

from typing import List, Protocol, Sequence

import httpx


class EmbeddingProvider(Protocol):
    """Backend-neutral embedding contract used by repository retrieval."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class OpenAICompatibleEmbeddingProvider:
    """Call an OpenAI-compatible `/embeddings` endpoint, including LLM Gateway."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not model:
            raise ValueError("embedding model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        rows = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in rows]
        if len(vectors) != len(texts):
            raise ValueError("embedding response count does not match request")
        return vectors
