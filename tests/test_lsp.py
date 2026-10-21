import io
from pathlib import Path

import pytest

from ai_harness.lsp import (
    DefinitionLocation,
    LSPDefinitionEnricher,
    LSPError,
    encode_message,
    read_message,
)
from ai_harness.models import SearchHit


def test_lsp_message_round_trip():
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
    encoded = encode_message(payload)

    assert read_message(io.BytesIO(encoded)) == payload


def test_lsp_message_requires_content_length():
    with pytest.raises(LSPError, match="Content-Length"):
        read_message(io.BytesIO(b"Content-Type: application/json\r\n\r\n{}"))


def test_lsp_definition_enricher_returns_only_repository_paths(tmp_path: Path, monkeypatch):
    source = tmp_path / "handler.py"
    target = tmp_path / "policy.py"
    source.write_text("def authorize():\n    return policy()\n", encoding="utf-8")
    target.write_text("def policy():\n    return True\n", encoding="utf-8")

    class FakeClient:
        def __init__(self, command, repository):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def open_document(self, relative_path, language_id, text):
            return None

        def definition(self, relative_path, line, character):
            return [
                DefinitionLocation(
                    uri=target.as_uri(),
                    start_line=0,
                    start_character=4,
                    end_line=0,
                    end_character=10,
                ),
                DefinitionLocation(
                    uri="file:///outside.py",
                    start_line=0,
                    start_character=0,
                    end_line=0,
                    end_character=1,
                ),
            ]

    monkeypatch.setattr("ai_harness.lsp.LSPClient", FakeClient)
    enricher = LSPDefinitionEnricher(["fake-lsp"], tmp_path)

    paths = enricher.related_paths(
        "authorize policy",
        [SearchHit(path="handler.py", line=1, text="authorize")],
    )

    assert paths == ["policy.py"]
