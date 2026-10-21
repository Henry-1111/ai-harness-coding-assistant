import io

import pytest

from ai_harness.lsp import LSPError, encode_message, read_message


def test_lsp_message_round_trip():
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
    encoded = encode_message(payload)

    assert read_message(io.BytesIO(encoded)) == payload


def test_lsp_message_requires_content_length():
    with pytest.raises(LSPError, match="Content-Length"):
        read_message(io.BytesIO(b"Content-Type: application/json\r\n\r\n{}"))
