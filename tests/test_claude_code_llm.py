from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness.llm.auth import OAuthCredential
from harness.llm.claude_code import (
    ClaudeCodeLLM,
    _build_payload,
    _default_user_agent,
    _parse_response,
)


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, *, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


class _Creds:
    def __init__(self):
        self.calls = []

    async def get_credential(self, *, force_refresh=False):
        self.calls.append(force_refresh)
        return OAuthCredential(
            provider="claude-code",
            access="fresh" if force_refresh else "stale",
            refresh="refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )


def test_build_payload_uses_anthropic_messages_shape():
    payload = _build_payload(
        model="claude-sonnet-4-5",
        system="sys",
        messages=[
            {"role": "system", "content": "ignored system"},
            {"role": "user", "content": "hi"},
        ],
        max_tokens=99,
        extra={"temperature": 0.2},
    )

    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["max_tokens"] == 99
    assert payload["temperature"] == 0.2
    assert payload["system"][0]["text"].startswith("x-anthropic-billing-header:")
    assert "Claude Code" in payload["system"][1]["text"]
    assert payload["system"][2]["text"] == "sys\n\nignored system"
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]


def test_parse_response_extracts_text_and_usage():
    text, usage = _parse_response(
        {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    )

    assert text == "ok"
    assert usage == {
        "tokens_in": 3,
        "tokens_out": 4,
        "total_tokens": 7,
        "provider": "claude-code",
    }


async def test_complete_posts_to_anthropic_and_refreshes_on_auth_error():
    client = _Client(
        [
            _Response(401, {"error": {"message": "expired"}}),
            _Response(
                200,
                {
                    "content": [{"type": "text", "text": '{"action":"finish"}'}],
                    "usage": {"input_tokens": 5, "output_tokens": 6},
                },
            ),
        ]
    )
    creds = _Creds()
    llm = ClaudeCodeLLM(
        model="claude-sonnet-4-6",
        credential_provider=creds,
        base_url="https://api.anthropic.com",
        http_client=client,
        user_agent="claude-cli/2.1.150 (external, cli)",
    )

    out = await llm.complete(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert out["text"] == '{"action":"finish"}'
    assert out["usage"]["total_tokens"] == 11
    assert creds.calls == [False, True]
    assert client.calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert client.calls[0]["json"]["model"] == "claude-sonnet-4-6"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer stale"
    assert client.calls[0]["headers"]["User-Agent"] == "claude-cli/2.1.150 (external, cli)"
    assert client.calls[0]["headers"]["anthropic-version"] == "2023-06-01"
    assert "oauth-2025-04-20" in client.calls[0]["headers"]["anthropic-beta"]
    assert client.calls[1]["headers"]["Authorization"] == "Bearer fresh"


def test_default_user_agent_can_be_overridden(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USER_AGENT", "custom-agent")
    assert _default_user_agent() == "custom-agent"
