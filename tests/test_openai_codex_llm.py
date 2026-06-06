from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness.llm.auth import OAuthCredential
from harness.llm.openai_codex import OpenAICodexLLM, _build_payload

# ── Mock streaming HTTP client ────────────────────────────────────────────────


class _StreamResponse:
    """Mock httpx-style streaming response."""

    def __init__(self, status_code: int, body: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.headers: dict[str, str] = {}

    async def aiter_lines(self):
        for line in self._body.splitlines():
            yield line

    async def aiter_bytes(self):
        yield self._body.encode()


class _StreamCtx:
    def __init__(self, response: _StreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _StreamResponse:
        return self._response

    async def __aexit__(self, *exc) -> None:
        pass


class _Client:
    """Mock httpx.AsyncClient that records calls and returns scripted responses."""

    def __init__(self, responses: list[_StreamResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def stream(self, method: str, url: str, *, headers, json):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json})
        return _StreamCtx(self.responses.pop(0))


class _Creds:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    async def get_credential(self, *, force_refresh: bool = False) -> OAuthCredential:
        self.calls.append(force_refresh)
        return OAuthCredential(
            provider="openai-codex",
            access="fresh" if force_refresh else "stale",
            refresh="refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            account_id="acct_123",
        )


# ── Payload builder ───────────────────────────────────────────────────────────


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


# ── stream_complete: yields deltas incrementally ─────────────────────────────


async def test_stream_complete_yields_deltas_incrementally():
    body = (
        "event: response.output_text.delta\n"
        'data: {"delta":"he"}\n\n'
        "event: response.output_text.delta\n"
        'data: {"delta":"llo"}\n\n'
        "event: response.completed\n"
        'data: {"response":{"usage":{"input_tokens":3,"output_tokens":4}}}\n\n'
    )
    client = _Client([_StreamResponse(200, body)])
    llm = OpenAICodexLLM(
        model="gpt-5.5",
        credential_provider=_Creds(),
        base_url="https://chatgpt.com/backend-api",
        http_client=client,
    )

    chunks: list[str] = []
    async for delta in llm.stream_complete(
        system=None, messages=[{"role": "user", "content": "hi"}]
    ):
        chunks.append(delta)

    assert chunks == ["he", "llo"]
    assert llm.last_usage["tokens_in"] == 3
    assert llm.last_usage["tokens_out"] == 4
    assert llm.last_usage["total_tokens"] == 7
    assert client.calls[0]["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer stale"
    assert "max_output_tokens" not in client.calls[0]["json"]


async def test_stream_complete_filters_unsupported_max_output_tokens():
    body = (
        "event: response.output_text.delta\n"
        'data: {"delta":"ok"}\n\n'
        "event: response.completed\n"
        'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    )
    client = _Client([_StreamResponse(200, body)])
    llm = OpenAICodexLLM(
        model="gpt-5.5",
        credential_provider=_Creds(),
        base_url="https://chatgpt.com/backend-api",
        http_client=client,
    )

    [delta async for delta in llm.stream_complete(system=None, messages=[], max_output_tokens=123)]

    assert "max_output_tokens" not in client.calls[0]["json"]


async def test_stream_complete_refreshes_creds_on_401():
    body = (
        "event: response.output_text.delta\n"
        'data: {"delta":"ok"}\n\n'
        "event: response.completed\n"
        'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    )
    client = _Client([_StreamResponse(401, ""), _StreamResponse(200, body)])
    creds = _Creds()
    llm = OpenAICodexLLM(
        model="gpt-5.5",
        credential_provider=creds,
        base_url="https://chatgpt.com/backend-api",
        http_client=client,
    )

    chunks: list[str] = []
    async for delta in llm.stream_complete(
        system=None, messages=[{"role": "user", "content": "hi"}]
    ):
        chunks.append(delta)

    assert chunks == ["ok"]
    assert creds.calls == [False, True]  # initial attempt, then force_refresh
    assert client.calls[0]["headers"]["Authorization"] == "Bearer stale"
    assert client.calls[1]["headers"]["Authorization"] == "Bearer fresh"


# ── complete: collects the stream ─────────────────────────────────────────────


async def test_complete_collects_streamed_deltas():
    body = (
        "event: response.output_text.delta\n"
        'data: {"delta":"{\\"action\\":\\"finish\\"}"}\n\n'
        "event: response.completed\n"
        'data: {"response":{"usage":{"input_tokens":5,"output_tokens":6}}}\n\n'
    )
    client = _Client([_StreamResponse(401, ""), _StreamResponse(200, body)])
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
    assert client.calls[0]["headers"]["Authorization"] == "Bearer stale"
    assert client.calls[1]["headers"]["Authorization"] == "Bearer fresh"
    assert client.calls[1]["headers"]["chatgpt-account-id"] == "acct_123"
    assert "max_output_tokens" not in client.calls[1]["json"]


async def test_complete_filters_unsupported_max_output_tokens():
    body = (
        "event: response.output_text.delta\n"
        'data: {"delta":"ok"}\n\n'
        "event: response.completed\n"
        'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    )
    client = _Client([_StreamResponse(200, body)])
    llm = OpenAICodexLLM(
        model="gpt-5.5",
        credential_provider=_Creds(),
        base_url="https://chatgpt.com/backend-api",
        http_client=client,
    )

    await llm.complete(system=None, messages=[], max_output_tokens=234)

    assert "max_output_tokens" not in client.calls[0]["json"]


async def test_complete_falls_back_to_final_payload_text():
    """If no delta events are sent but the final payload has output_text, use it."""
    body = (
        "event: response.completed\n"
        'data: {"response":{"output_text":"only-final",'
        '"usage":{"input_tokens":2,"output_tokens":3}}}\n\n'
    )
    client = _Client([_StreamResponse(200, body)])
    llm = OpenAICodexLLM(
        model="gpt-5.5",
        credential_provider=_Creds(),
        base_url="https://chatgpt.com/backend-api",
        http_client=client,
    )

    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])

    assert out["text"] == "only-final"
    assert out["usage"]["total_tokens"] == 5
