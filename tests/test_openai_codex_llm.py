from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness.llm.auth import OAuthCredential
from harness.llm.openai_codex import (
    OpenAICodexLLM,
    _build_payload,
    _parse_response,
    _parse_sse_response,
)


class _Response:
    def __init__(self, status_code: int, payload: dict, *, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


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
            provider="openai-codex",
            access="fresh" if force_refresh else "stale",
            refresh="refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            account_id="acct_123",
        )


def test_build_payload_uses_responses_shape():
    payload = _build_payload(
        model="gpt-5.5",
        system="sys",
        messages=[
            {"role": "system", "content": "extra sys"},
            {"role": "user", "content": "hi"},
        ],
        extra={"service_tier": "priority"},
    )

    assert payload["model"] == "gpt-5.5"
    assert payload["instructions"] == "sys\n\nextra sys"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["tools"] == []
    assert payload["tool_choice"] == "none"
    assert payload["service_tier"] == "priority"
    assert payload["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        },
    ]


def test_parse_response_accepts_output_text():
    text, usage = _parse_response(
        {
            "output_text": "ok",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    )
    assert text == "ok"
    assert usage["tokens_in"] == 3
    assert usage["tokens_out"] == 4
    assert usage["provider"] == "openai-codex"


def test_parse_response_accepts_output_blocks():
    text, usage = _parse_response(
        {
            "output": [
                {"content": [{"type": "output_text", "text": "he"}, {"text": "llo"}]},
            ]
        }
    )
    assert text == "hello"
    assert usage == {}


def test_parse_sse_response_accepts_codex_events():
    response = _Response(
        200,
        {},
        text=(
            "event: response.output_text.delta\n"
            'data: {"delta":"he"}\n\n'
            "event: response.output_text.delta\n"
            'data: {"delta":"llo"}\n\n'
            "event: response.completed\n"
            'data: {"response":{"usage":{"input_tokens":3,"output_tokens":4}}}\n\n'
        ),
    )

    text, usage = _parse_sse_response(response)

    assert text == "hello"
    assert usage["tokens_in"] == 3
    assert usage["tokens_out"] == 4


async def test_complete_posts_to_codex_backend_and_refreshes_on_auth_error():
    client = _Client(
        [
            _Response(401, {"error": "expired"}),
            _Response(
                200,
                {},
                text=(
                    "event: response.output_text.delta\n"
                    'data: {"delta":"{\\"action\\":\\"finish\\"}"}\n\n'
                    "event: response.completed\n"
                    'data: {"response":{"usage":{"input_tokens":5,"output_tokens":6}}}\n\n'
                ),
            ),
        ]
    )
    creds = _Creds()
    llm = OpenAICodexLLM(
        model="gpt-5.5",
        credential_provider=creds,
        base_url="https://chatgpt.com/backend-api",
        http_client=client,
    )

    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])

    assert out["text"] == '{"action":"finish"}'
    assert out["usage"]["total_tokens"] == 11
    assert creds.calls == [False, True]
    assert client.calls[0]["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer stale"
    assert client.calls[1]["headers"]["Authorization"] == "Bearer fresh"
    assert client.calls[1]["headers"]["chatgpt-account-id"] == "acct_123"
