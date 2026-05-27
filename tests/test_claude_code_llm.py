from __future__ import annotations

from datetime import datetime, timedelta, timezone

import harness.llm.claude_code as cc_mod
from harness.llm.auth import OAuthCredential
from harness.llm.claude_code import ClaudeCodeLLM, _build_payload, _default_user_agent

# ── Mock streaming HTTP client ────────────────────────────────────────────────


class _StreamResponse:
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
            provider="claude-code",
            access="fresh" if force_refresh else "stale",
            refresh="refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )


# ── Streaming SSE body helper ─────────────────────────────────────────────────


def _sse(events: list[dict]) -> str:
    """Format a list of Anthropic SSE event objects as a body string."""
    import json

    out: list[str] = []
    for evt in events:
        evt_type = evt.get("type", "message")
        out.append(f"event: {evt_type}")
        out.append(f"data: {json.dumps(evt)}")
        out.append("")
    return "\n".join(out) + "\n"


# ── Payload + headers ─────────────────────────────────────────────────────────


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


def test_default_user_agent_can_be_overridden(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USER_AGENT", "custom-agent")
    assert _default_user_agent() == "custom-agent"


def test_cc_version_resolves_from_env(monkeypatch):
    """The billing header reflects CLAUDE_CODE_VERSION when set."""
    monkeypatch.setenv("CLAUDE_CODE_VERSION", "2.5.0-test")
    payload = _build_payload(
        model="claude-sonnet-4-5",
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=99,
        extra={},
    )
    billing = payload["system"][0]["text"]
    assert "cc_version=2.5.0-test" in billing


def test_cc_version_falls_back_to_installed_cli(monkeypatch):
    """When env is unset, the resolver consults the installed CLI; cached."""
    monkeypatch.delenv("CLAUDE_CODE_VERSION", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USER_AGENT", raising=False)
    # Reset the module-level cache, then stub the probe.
    monkeypatch.setattr(cc_mod, "_cached_cli_version", None)
    monkeypatch.setattr(cc_mod, "_cli_version_probed", False)
    monkeypatch.setattr(cc_mod, "_installed_claude_version", lambda: "9.9.9")

    payload = _build_payload(
        model="claude-sonnet-4-5",
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=99,
        extra={},
    )
    billing = payload["system"][0]["text"]
    assert "cc_version=9.9.9" in billing
    assert _default_user_agent() == "claude-cli/9.9.9 (external, cli)"


def test_cc_version_unknown_when_cli_absent_and_no_env(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_VERSION", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USER_AGENT", raising=False)
    monkeypatch.setattr(cc_mod, "_cached_cli_version", None)
    monkeypatch.setattr(cc_mod, "_cli_version_probed", False)
    monkeypatch.setattr(cc_mod, "_installed_claude_version", lambda: None)

    payload = _build_payload(
        model="claude-sonnet-4-5",
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=99,
        extra={},
    )
    billing = payload["system"][0]["text"]
    assert "cc_version=unknown" in billing


# ── stream_complete: yields deltas, captures usage ───────────────────────────


async def test_stream_complete_yields_deltas_incrementally():
    body = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_start"},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "he"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "llo"}},
            {"type": "content_block_stop"},
            {"type": "message_delta", "usage": {"output_tokens": 6}},
            {"type": "message_stop"},
        ]
    )
    client = _Client([_StreamResponse(200, body)])
    llm = ClaudeCodeLLM(
        model="claude-sonnet-4-6",
        credential_provider=_Creds(),
        base_url="https://api.anthropic.com",
        http_client=client,
        user_agent="claude-cli/2.1.150 (external, cli)",
    )

    chunks: list[str] = []
    async for delta in llm.stream_complete(
        system="sys", messages=[{"role": "user", "content": "hi"}]
    ):
        chunks.append(delta)

    assert chunks == ["he", "llo"]
    assert llm.last_usage == {
        "tokens_in": 5,
        "tokens_out": 6,
        "total_tokens": 11,
        "provider": "claude-code",
    }
    assert client.calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert client.calls[0]["json"]["stream"] is True
    assert client.calls[0]["headers"]["User-Agent"] == "claude-cli/2.1.150 (external, cli)"
    assert client.calls[0]["headers"]["anthropic-version"] == "2023-06-01"
    assert "oauth-2025-04-20" in client.calls[0]["headers"]["anthropic-beta"]


async def test_stream_complete_refreshes_creds_on_401():
    body = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_delta", "usage": {"output_tokens": 1}},
        ]
    )
    client = _Client([_StreamResponse(401, ""), _StreamResponse(200, body)])
    creds = _Creds()
    llm = ClaudeCodeLLM(
        model="claude-sonnet-4-6",
        credential_provider=creds,
        base_url="https://api.anthropic.com",
        http_client=client,
        user_agent="claude-cli/2.1.150 (external, cli)",
    )

    chunks: list[str] = []
    async for delta in llm.stream_complete(
        system="sys", messages=[{"role": "user", "content": "hi"}]
    ):
        chunks.append(delta)

    assert chunks == ["ok"]
    assert creds.calls == [False, True]
    assert client.calls[0]["headers"]["Authorization"] == "Bearer stale"
    assert client.calls[1]["headers"]["Authorization"] == "Bearer fresh"


# ── complete: collects the streamed deltas ────────────────────────────────────


async def test_complete_collects_streamed_deltas():
    body = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": '{"action":"finish"}'},
            },
            {"type": "message_delta", "usage": {"output_tokens": 6}},
        ]
    )
    client = _Client([_StreamResponse(401, ""), _StreamResponse(200, body)])
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
    assert client.calls[1]["json"]["model"] == "claude-sonnet-4-6"
    assert client.calls[1]["json"]["stream"] is True
