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
    embedding_model: str
    lsp_command: str
    sandbox_mode: str
    sandbox_image: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            data_directory=Path(os.getenv("AI_HARNESS_DATA_DIR", ".ai-harness")),
            model_base_url=os.getenv("AI_HARNESS_MODEL_BASE_URL", "https://api.openai.com/v1"),
            model_api_key=os.getenv("AI_HARNESS_MODEL_API_KEY", ""),
            model_name=os.getenv("AI_HARNESS_MODEL", "gpt-4.1-mini"),
            model_timeout_seconds=float(os.getenv("AI_HARNESS_MODEL_TIMEOUT", "60")),
            embedding_model=os.getenv("AI_HARNESS_EMBEDDING_MODEL", ""),
            lsp_command=os.getenv("AI_HARNESS_LSP_COMMAND", ""),
            sandbox_mode=os.getenv("AI_HARNESS_SANDBOX_MODE", "docker"),
            sandbox_image=os.getenv("AI_HARNESS_SANDBOX_IMAGE", "python:3.12-slim"),
        )
