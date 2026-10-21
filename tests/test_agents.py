import json

from ai_harness.agents import OpenAICompatibleModel


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Gateway response",
                                "edits": [],
                            }
                        )
                    }
                }
            ]
        }


def test_openai_adapter_matches_llm_gateway_contract(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("ai_harness.agents.httpx.post", fake_post)
    model = OpenAICompatibleModel(
        "http://127.0.0.1:8000/v1/",
        "llmgw-tenant-key",
        "gpt-4.1-mini",
        timeout_seconds=15,
    )

    result = model.complete_json(system="return JSON", payload={"goal": "fix bug"})

    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer llmgw-tenant-key"
    assert captured["body"]["model"] == "gpt-4.1-mini"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 15
    assert result["summary"] == "Gateway response"
