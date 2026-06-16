"""
AnthropicLLM adapter tests with mocked anthropic SDK (no API calls).

We monkey-patch anthropic.AsyncAnthropic so the adapter's request shape is
verified without hitting the network. Both the non-streaming `.messages.create()`
and the streaming `.messages.stream()` context-manager paths are stubbed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")
from harness.llm.anthropic import (  # noqa: E402
    AnthropicLLM,
    _build_messages,
    _system_blocks,
)

# ── Fake Anthropic SDK surface ────────────────────────────────────────────────


class _FakeUsage:
    def __init__(
        self,
        input_tokens: int = 10,
        output_tokens: int = 20,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _FakeContentBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(
        self,
        content: list[_FakeContentBlock],
        *,
        model: str = "claude-test",
        usage: _FakeUsage | None = None,
    ):
        self.content = content
        self.model = model
        self.usage = usage or _FakeUsage()


class _FakeStreamCtx:
    """Mimics `async with client.messages.stream(...) as stream`."""

    def __init__(self, tokens: list[str], *, final: _FakeMessage):
        self._tokens = tokens
        self._final = final
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    @property
    def text_stream(self):
        return self._aiter_tokens()

    async def _aiter_tokens(self):
        for tok in self._tokens:
            yield tok

    async def get_final_message(self) -> _FakeMessage:
        return self._final


class _FakeMessages:
    """Stand-in for client.messages."""

    def __init__(self):
        self.calls: list[dict] = []
        self.next_response: _FakeMessage | None = None
        self.next_stream_tokens: list[str] = []
        self.next_stream_final: _FakeMessage | None = None

    async def create(self, **kwargs) -> _FakeMessage:
        self.calls.append({"path": "create", **kwargs})
        return self.next_response

    def stream(self, **kwargs) -> _FakeStreamCtx:
        self.calls.append({"path": "stream", **kwargs})
        return _FakeStreamCtx(
            self.next_stream_tokens,
            final=self.next_stream_final,
        )


def _build(monkeypatch, **kwargs):
    """Build an AnthropicLLM whose client.messages is fully faked."""
    fake_messages = _FakeMessages()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = fake_messages

    monkeypatch.setattr("anthropic.AsyncAnthropic", _FakeClient)
    return AnthropicLLM(model="claude-test", **kwargs), fake_messages


# ── Pure helpers ──────────────────────────────────────────────────────────────


def test_system_blocks_with_caching():
    blocks = _system_blocks("you are helpful", prompt_caching=True)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "you are helpful"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_system_blocks_without_caching():
    blocks = _system_blocks("you are helpful", prompt_caching=False)
    assert len(blocks) == 1
    assert "cache_control" not in blocks[0]


def test_system_blocks_empty():
    assert _system_blocks(None, prompt_caching=True) == []
    assert _system_blocks("", prompt_caching=True) == []


def test_build_messages_converts_to_content_block_list():
    msgs = _build_messages(
        [{"role": "user", "content": "hello"}],
        prompt_caching=False,
    )
    assert msgs == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_build_messages_drops_system_role():
    msgs = _build_messages(
        [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "hi"},
        ],
        prompt_caching=False,
    )
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_build_messages_adds_cache_control_to_last_user():
    msgs = _build_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "second"},
        ],
        prompt_caching=True,
    )
    assert msgs[-1]["content"][0].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in msgs[0]["content"][0]


def test_build_messages_no_cache_control_when_disabled():
    msgs = _build_messages(
        [{"role": "user", "content": "hi"}],
        prompt_caching=False,
    )
    assert "cache_control" not in msgs[-1]["content"][0]


# ── complete() ────────────────────────────────────────────────────────────────


async def test_complete_returns_text(monkeypatch):
    llm, messages = _build(monkeypatch)
    messages.next_response = _FakeMessage(
        [_FakeContentBlock("Hello world")],
        model="claude-test",
        usage=_FakeUsage(input_tokens=5, output_tokens=10),
    )

    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])

    assert out["text"] == "Hello world"
    assert out["usage"]["tokens_in"] == 5
    assert out["usage"]["tokens_out"] == 10
    assert out["usage"]["model"] == "claude-test"
    assert "cost_usd" not in out["usage"]


async def test_complete_sets_last_usage(monkeypatch):
    llm, messages = _build(monkeypatch)
    messages.next_response = _FakeMessage(
        [_FakeContentBlock("hi")],
        usage=_FakeUsage(input_tokens=7, output_tokens=3),
    )

    await llm.complete(system=None, messages=[{"role": "user", "content": "x"}])
    assert llm.last_usage["tokens_in"] == 7
    assert llm.last_usage["tokens_out"] == 3


async def test_complete_sends_system_as_block_list_with_caching(monkeypatch):
    llm, messages = _build(monkeypatch, prompt_caching=True)
    messages.next_response = _FakeMessage([_FakeContentBlock("ok")])

    await llm.complete(
        system="be a helpful assistant",
        messages=[{"role": "user", "content": "hi"}],
    )

    call = messages.calls[0]
    assert call["path"] == "create"
    system = call["system"]
    assert isinstance(system, list)
    assert system[0]["type"] == "text"
    assert system[0]["text"] == "be a helpful assistant"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


async def test_complete_sends_system_as_plain_block_without_caching(monkeypatch):
    llm, messages = _build(monkeypatch, prompt_caching=False)
    messages.next_response = _FakeMessage([_FakeContentBlock("ok")])

    await llm.complete(
        system="be a helpful assistant",
        messages=[{"role": "user", "content": "hi"}],
    )

    call = messages.calls[0]
    system = call["system"]
    assert isinstance(system, list)
    assert "cache_control" not in system[0]


async def test_complete_no_system_omits_system_param(monkeypatch):
    llm, messages = _build(monkeypatch)
    messages.next_response = _FakeMessage([_FakeContentBlock("ok")])

    await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])

    call = messages.calls[0]
    assert "system" not in call


# ── last_usage includes cache fields ─────────────────────────────────────────


async def test_complete_last_usage_has_cache_fields(monkeypatch):
    llm, messages = _build(monkeypatch)
    messages.next_response = _FakeMessage(
        [_FakeContentBlock("ok")],
        usage=_FakeUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=20,
        ),
    )

    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])

    assert out["usage"]["cache_read_tokens"] == 80
    assert out["usage"]["cache_creation_tokens"] == 20
    assert llm.last_usage["cache_read_tokens"] == 80
    assert llm.last_usage["cache_creation_tokens"] == 20


# ── stream_complete() ─────────────────────────────────────────────────────────


async def test_stream_complete_yields_tokens(monkeypatch):
    llm, messages = _build(monkeypatch)
    messages.next_stream_tokens = ["He", "llo", " world"]
    messages.next_stream_final = _FakeMessage(
        [],
        usage=_FakeUsage(input_tokens=4, output_tokens=3),
    )

    tokens = [
        t
        async for t in llm.stream_complete(
            system=None, messages=[{"role": "user", "content": "hi"}]
        )
    ]

    assert tokens == ["He", "llo", " world"]


async def test_stream_complete_sets_last_usage(monkeypatch):
    llm, messages = _build(monkeypatch)
    messages.next_stream_tokens = ["ok"]
    messages.next_stream_final = _FakeMessage(
        [],
        usage=_FakeUsage(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=8,
            cache_creation_input_tokens=2,
        ),
    )

    [
        t
        async for t in llm.stream_complete(
            system=None, messages=[{"role": "user", "content": "hi"}]
        )
    ]

    assert llm.last_usage["tokens_in"] == 10
    assert llm.last_usage["tokens_out"] == 5
    assert llm.last_usage["cache_read_tokens"] == 8
    assert llm.last_usage["cache_creation_tokens"] == 2


async def test_stream_complete_caching_enabled_wraps_system(monkeypatch):
    llm, messages = _build(monkeypatch, prompt_caching=True)
    messages.next_stream_tokens = ["hi"]
    messages.next_stream_final = _FakeMessage([], usage=_FakeUsage())

    [
        t
        async for t in llm.stream_complete(
            system="my system",
            messages=[{"role": "user", "content": "go"}],
        )
    ]

    call = messages.calls[0]
    assert call["path"] == "stream"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_stream_complete_caching_disabled(monkeypatch):
    llm, messages = _build(monkeypatch, prompt_caching=False)
    messages.next_stream_tokens = ["hi"]
    messages.next_stream_final = _FakeMessage([], usage=_FakeUsage())

    [
        t
        async for t in llm.stream_complete(
            system="my system",
            messages=[{"role": "user", "content": "go"}],
        )
    ]

    call = messages.calls[0]
    assert "cache_control" not in call["system"][0]


async def test_json_reminder_appended_on_stream_complete(monkeypatch):
    """response_format=json_object appends JSON reminder to last user message."""
    llm, messages = _build(monkeypatch)
    messages.next_stream_tokens = ['{"thought": "ok"}']
    messages.next_stream_final = _FakeMessage([], usage=_FakeUsage())

    tokens = [
        t
        async for t in llm.stream_complete(
            system=None,
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
    ]

    assert "".join(tokens) == '{"thought": "ok"}'
    call = messages.calls[0]
    last_msg = call["messages"][-1]
    assert last_msg["role"] == "user"
    assert "Respond with a JSON object only." in last_msg["content"][-1]["text"]


async def test_json_reminder_appended_on_complete(monkeypatch):
    """response_format=json_object appends JSON reminder to last user message."""
    llm, messages = _build(monkeypatch)
    messages.next_response = _FakeMessage(
        [_FakeContentBlock('{"thought": "ok"}')],
        usage=_FakeUsage(),
    )

    out = await llm.complete(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    assert out["text"] == '{"thought": "ok"}'
    call = messages.calls[0]
    last_msg = call["messages"][-1]
    assert last_msg["role"] == "user"
    assert "Respond with a JSON object only." in last_msg["content"][-1]["text"]


async def test_no_reminder_without_json_mode(monkeypatch):
    """Without response_format, no JSON reminder is injected."""
    llm, messages = _build(monkeypatch)
    messages.next_stream_tokens = ["hello"]
    messages.next_stream_final = _FakeMessage([], usage=_FakeUsage())

    tokens = [
        t
        async for t in llm.stream_complete(
            system=None, messages=[{"role": "user", "content": "hi"}]
        )
    ]

    assert tokens == ["hello"]
    call = messages.calls[0]
    assert "Respond with a JSON object only." not in call["messages"][-1]["content"][-1]["text"]


# ── cost_fn and BudgetGuard ───────────────────────────────────────────────────


async def test_cost_fn_applied_to_usage(monkeypatch):
    def my_cost(u):
        return u["tokens_in"] * 1e-6 + u["tokens_out"] * 3e-6

    llm, messages = _build(monkeypatch, cost_fn=my_cost)
    messages.next_response = _FakeMessage(
        [_FakeContentBlock("ok")],
        usage=_FakeUsage(input_tokens=100, output_tokens=50),
    )

    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])

    expected = 100 * 1e-6 + 50 * 3e-6
    assert out["usage"]["cost_usd"] == pytest.approx(expected)


async def test_cost_fn_exception_swallowed(monkeypatch):
    def broken(u):
        raise ValueError("pricing unavailable")

    llm, messages = _build(monkeypatch, cost_fn=broken)
    messages.next_response = _FakeMessage([_FakeContentBlock("ok")])

    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])
    assert "cost_usd" not in out["usage"]
    assert out["text"] == "ok"


async def test_stream_complete_falls_back_to_final_message_when_text_stream_empty(monkeypatch):
    """Bedrock compatibility: text_stream yields nothing but final message has content."""
    llm, messages = _build(monkeypatch)
    messages.next_stream_tokens = []  # simulate Bedrock empty text_stream
    messages.next_stream_final = _FakeMessage(
        [_FakeContentBlock('{"thought": "ok", "action": "finish"}')],
        usage=_FakeUsage(input_tokens=10, output_tokens=5),
    )

    tokens = [
        t
        async for t in llm.stream_complete(
            system=None, messages=[{"role": "user", "content": "hi"}]
        )
    ]

    assert tokens == ['{"thought": "ok", "action": "finish"}']
    assert llm.last_usage["tokens_in"] == 10


async def test_set_budget_routes_cost_to_guard(monkeypatch):
    from harness.runtime import BudgetGuard, GuardrailConfig

    llm, messages = _build(monkeypatch, cost_fn=lambda u: 0.0125)
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0))
    llm.set_budget(guard)

    messages.next_response = _FakeMessage(
        [_FakeContentBlock("ok")],
        usage=_FakeUsage(input_tokens=10, output_tokens=5),
    )
    await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])
    assert guard.cost == pytest.approx(0.0125)
