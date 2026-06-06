"""
Anthropic LLM adapter (direct API key, no OAuth).

Implements the harness LLM client contract:
  - async def complete(system, messages, **kwargs) -> dict
  - async def stream_complete(system, messages) -> AsyncGenerator[str, None]

Prompt caching
--------------
Enabled by default (`prompt_caching=True`). When active:
  - The system prompt is sent as a content-block list with `cache_control`
    on the last block so Anthropic can cache the compiled KV state.
  - The last user message's text block also gets `cache_control` so
    multi-turn ReAct loops that share a common leading prefix cache cheaply.

Cache reads cost ~10% of normal input tokens. Callers that pass a `cost_fn`
receive `cache_read_tokens` and `cache_creation_tokens` in the usage dict so
they can apply the correct per-tier pricing.

Usage tracking
--------------
`last_usage` is populated after every call::

    {
        "tokens_in": int,                 # non-cached input tokens
        "tokens_out": int,                # output tokens
        "cache_read_tokens": int,         # tokens served from cache
        "cache_creation_tokens": int,     # tokens written to cache
        "model": str,                     # model id echoed from response
    }

Cost tracking
-------------
An optional `cost_fn(usage) -> float` may be supplied to convert the usage
dict to dollars. This is handy for callers that know the per-model pricing
schedule. When `set_budget(guard)` is called (typically by AgentRuntime),
the adapter forwards computed costs to the guard's `add_cost()` method.

Install:
    pip install -e ".[anthropic]"

Usage:
    from harness.llm.anthropic import AnthropicLLM
    llm = AnthropicLLM(model="claude-sonnet-4-6")  # reads ANTHROPIC_API_KEY
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

logger = logging.getLogger(__name__)


class AnthropicLLM:
    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,  # falls back to ANTHROPIC_API_KEY env
        # Generous default — matches OpenAILLM's max_completion_tokens=4096.
        # ReAct ``thought`` fields + finish answers were getting clipped at
        # 1024 once observations grew large; 4096 leaves comfortable headroom
        # while staying below Claude 3.5 / 3.7's per-call output ceilings.
        max_tokens: int = 4096,
        cost_fn: Callable[[dict], float] | None = None,
        prompt_caching: bool = True,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                'anthropic package not installed. Run: pip install -e ".[anthropic]"'
            ) from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._cost_fn = cost_fn
        self._prompt_caching = prompt_caching
        self._budget: Any = None
        # Populated after every successful call; streaming callers read it here.
        self.last_usage: dict | None = None

    def set_budget(self, guard: Any) -> None:
        """Inject a BudgetGuard; AgentRuntime calls this at the start of each run."""
        self._budget = guard

    # ── Non-streaming ──────────────────────────────────────────────────────────

    async def complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> dict:
        max_tokens = int(kwargs.pop("max_tokens", self._max_tokens))
        sys_blocks = _system_blocks(system, prompt_caching=self._prompt_caching)
        built_messages = _build_messages(messages, prompt_caching=self._prompt_caching)

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": built_messages,
        }
        if sys_blocks:
            request["system"] = sys_blocks

        resp = await self._client.messages.create(**request)
        usage = _extract_usage(resp.usage, resp.model or self._model)
        cost = _compute_cost(usage, self._cost_fn)
        if cost is not None:
            usage["cost_usd"] = cost
        self._record_usage(usage, source=source)
        self.last_usage = usage

        text = _collect_text(resp.content)
        return {"text": text, "usage": usage}

    # ── Streaming ──────────────────────────────────────────────────────────────

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        **_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        # ``_kwargs`` swallows OpenAI-style hints like ``response_format`` —
        # Anthropic doesn't expose an equivalent (structure is enforced via
        # prefill or system prompt). Accept and ignore so a caller wiring
        # the same ReAct prompt at both adapters doesn't crash here.
        sys_blocks = _system_blocks(system, prompt_caching=self._prompt_caching)
        built_messages = _build_messages(messages, prompt_caching=self._prompt_caching)

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": built_messages,
        }
        if sys_blocks:
            request["system"] = sys_blocks

        async with self._client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                yield text

            final = await stream.get_final_message()
            usage = _extract_usage(final.usage, final.model or self._model)
            cost = _compute_cost(usage, self._cost_fn)
            if cost is not None:
                usage["cost_usd"] = cost
            self._record_usage(usage, source=source)
            self.last_usage = usage

    # ── Internals ─────────────────────────────────────────────────────────────

    def _record_usage(self, usage: dict, *, source: str | None) -> None:
        """Forward usage to the budget guard.

        Token count for budget purposes is the total input that hit the wire
        — non-cached + cache-creation + cache-read — so token caps reflect
        real wall-clock consumption regardless of cache hit rate. Cost
        (which respects cache pricing via ``cost_fn``) is reported when
        known.
        """
        guard = self._budget
        if not guard:
            return
        tokens_in = (
            int(usage.get("tokens_in") or 0)
            + int(usage.get("cache_read_tokens") or 0)
            + int(usage.get("cache_creation_tokens") or 0)
        )
        tokens_out = int(usage.get("tokens_out") or 0)
        if (tokens_in or tokens_out) and hasattr(guard, "add_tokens"):
            guard.add_tokens(tokens_in, tokens_out, source=source)
        cost = usage.get("cost_usd")
        if cost and cost > 0:
            guard.add_cost(cost, source=source)


# ── Module-level helpers ──────────────────────────────────────────────────────


def _system_blocks(system: str | None, *, prompt_caching: bool) -> list[dict[str, Any]]:
    """Return the system param as a content-block list (or empty list for no system)."""
    if not system:
        return []
    block: dict[str, Any] = {"type": "text", "text": system}
    if prompt_caching:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _build_messages(messages: list[dict], *, prompt_caching: bool) -> list[dict[str, Any]]:
    """Convert harness message dicts to Anthropic message format.

    System-role messages are silently dropped (callers should pass them via
    the `system` parameter). The last user message gets `cache_control` when
    prompt_caching is enabled.
    """
    built: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue  # consumed by caller as the system param
        if role not in {"user", "assistant"}:
            role = "user"
        content = msg.get("content", "")
        built.append(
            {
                "role": role,
                "content": [{"type": "text", "text": str(content)}],
            }
        )

    if prompt_caching:
        _apply_last_user_cache_control(built)

    return built


def _apply_last_user_cache_control(messages: list[dict]) -> None:
    """Add cache_control to the last user message's single text block."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text":
            content[0]["cache_control"] = {"type": "ephemeral"}
        break


def _extract_usage(usage: Any, model: str) -> dict:
    """Build the standard harness usage dict from an Anthropic usage object."""
    return {
        "tokens_in": getattr(usage, "input_tokens", 0),
        "tokens_out": getattr(usage, "output_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "model": model,
    }


def _collect_text(content: Any) -> str:
    """Extract plain text from an Anthropic response content list."""
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _compute_cost(usage: dict, cost_fn: Callable[[dict], float] | None) -> float | None:
    if cost_fn is None:
        return None
    try:
        return float(cost_fn(usage))
    except Exception as e:
        logger.warning("cost_fn raised: %s — skipping cost for this call", e)
        return None
