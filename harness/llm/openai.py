"""
OpenAI LLM adapter.

Implements the harness's LLM client contract:
  - async def complete(system, messages, **kwargs) -> dict
  - async def stream_complete(system, messages) -> AsyncGenerator[str, None]

Cost tracking
-------------
The adapter always captures provider-reported `usage` (prompt_tokens,
completion_tokens, model) from every call. Dollars are optional and come from
one of three sources, in precedence order:

  1. Gateway header — if the OpenAI client is pointed at a proxy (LiteLLM,
     Helicone, etc) that returns a per-request cost in a response header.
     Detected: `x-litellm-response-cost`, `x-cost-usd`, `x-helicone-cost-usd`.
  2. `cost_fn(usage) -> float` — caller-supplied. Receives the usage dict
     ({tokens_in, tokens_out, model, ...}) and returns dollars.
  3. Neither — only token counts get reported; `cost_usd` is omitted.

When `set_budget(guard)` is called (typically by AgentRuntime per-run), the
adapter forwards `cost_usd` to `guard.add_cost()` so the BudgetGuard ceiling
can fire. With no budget set, the adapter is purely observational.

The harness uses `system=None` for the agent ReAct path (the system prompt
sits inside `messages` as the first message). For orchestrator/memory paths
(planning, synthesis, extraction, summarization) `system` is a string. This
adapter prepends it as a "system" role message when provided.

Install:
    pip install -e ".[openai]"

Usage:
    from harness.llm.openai import OpenAILLM
    llm = OpenAILLM(model="gpt-4o-mini")              # reads OPENAI_API_KEY from env
    # or, routing through a gateway that emits cost headers:
    llm = OpenAILLM(model="gpt-4o-mini", base_url="https://my-litellm/v1")
    # or, with a local pricing function:
    def my_pricing(usage):
        rate_in, rate_out = 0.15e-6, 0.60e-6   # gpt-4o-mini, per token
        return usage["tokens_in"] * rate_in + usage["tokens_out"] * rate_out
    llm = OpenAILLM(model="gpt-4o-mini", cost_fn=my_pricing)
    runtime = AgentRuntime(..., llm=llm)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Response headers commonly emitted by proxy gateways for per-request cost.
_GATEWAY_COST_HEADERS = (
    "x-litellm-response-cost",
    "x-cost-usd",
    "x-helicone-cost-usd",
)

# Sentinel distinguishing "caller didn't pass max_completion_tokens" from
# "caller explicitly passed None" — the latter opts out of the cap entirely.
_SENTINEL: Any = object()


# Known OpenAI model context windows (input + output combined). OpenAI's
# ``/v1/models`` endpoint does NOT expose this — we maintain a table and
# fall back conservatively for unknowns. Users running new models should
# pass ``context_window=N`` to the constructor explicitly.
#
# Keep entries ordered with the most-specific prefixes first so the prefix
# match in ``_lookup`` resolves the right cell.
_OPENAI_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.4-mini": 128_000,
    "gpt-5.4": 128_000,
    "gpt-5": 256_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1-mini": 128_000,
    "o1": 200_000,
}
# Conservative ceiling for unknown models. Better to under-utilise context
# than over-promise and hit a 413 in the middle of a ReAct loop.
_OPENAI_CONTEXT_WINDOW_FALLBACK = 8_000
# Slack reserved at the top of the context window for tool args / variance
# in token counting (this codebase counts chars/4; tiktoken differs ~10%).
_OPENAI_TOKEN_BUDGET_SAFETY = 512


class OpenAILLM:
    def __init__(
        self,
        *,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,  # falls back to OPENAI_API_KEY env
        base_url: str | None = None,  # set when routing through a gateway
        request_timeout_seconds: float = 60.0,
        cost_fn: Callable[[dict], float] | None = None,
        # Generous default so long ReAct ``thought`` fields + structured
        # JSON ``finish`` answers don't get clipped mid-stream — the
        # default-None behaviour was relying on the model's own ceiling,
        # which truncated JSON-mode responses inside ``BaseAgent``'s loop.
        # Per-call override available via ``complete(..., max_completion_tokens=N)``.
        max_completion_tokens: int | None = 4096,
        # Explicit context window override. When None (default), the
        # adapter consults ``_OPENAI_CONTEXT_WINDOWS`` and falls back to
        # 8K + a warning for models it doesn't recognise. Pass an int when
        # running a new model the table doesn't know yet.
        context_window: int | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                'openai package not installed. Run: pip install -e ".[openai]"'
            ) from e

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout_seconds,
        )
        self._model = model
        self._cost_fn = cost_fn
        self._max_completion_tokens = max_completion_tokens
        self._context_window = context_window or _lookup_openai_context_window(model)
        self._budget = None
        # Last observed usage dict. Populated after every successful call; useful
        # for streaming callers who can't read it from the return value.
        self.last_usage: dict | None = None

    @property
    def context_window(self) -> int:
        """Total context window (input + output) for the configured model."""
        return self._context_window

    @property
    def input_token_budget(self) -> int:
        """Tokens available for the prompt after reserving output capacity.

        Used by ``WorkingMemory`` to derive its eviction threshold —
        compacting at e.g. 80% of this value gives the LLM real headroom
        instead of forcing premature summarisation at a hardcoded 8K.
        """
        output_reserve = self._max_completion_tokens or 0
        return max(
            1024,  # never report a negative or near-zero budget
            self._context_window - output_reserve - _OPENAI_TOKEN_BUDGET_SAFETY,
        )

    def set_budget(self, guard: Any) -> None:
        """
        Inject a BudgetGuard so the adapter can call add_cost() on every call.
        AgentRuntime calls this with a fresh guard at the start of each run.
        """
        self._budget = guard

    # ── Non-streaming ─────────────────────────────────────────────────────────

    async def complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> dict:
        full_messages = _prepend_system(system, messages)
        request: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
        }
        # Pass through response_format only if the caller asked for it; OpenAI
        # supports {"type": "json_object"} to enforce strict JSON output.
        if "response_format" in kwargs:
            request["response_format"] = kwargs["response_format"]
        # Per-call override wins over the instance default; explicit ``None``
        # means "let the model use its own ceiling" so callers can opt out.
        max_completion_tokens = kwargs.get("max_completion_tokens", self._max_completion_tokens)
        if max_completion_tokens is not None:
            request["max_completion_tokens"] = max_completion_tokens

        # with_raw_response gives us response headers (for gateway cost detection)
        # plus the parsed body. Both succeed identically if the gateway header
        # isn't present — there's no extra cost vs the simple .create() path.
        raw = await self._client.chat.completions.with_raw_response.create(**request)
        resp = raw.parse()
        headers = _headers_dict(raw)
        usage = self._build_usage(resp, headers)
        self._record_usage(usage, source=source)
        self.last_usage = usage

        content = resp.choices[0].message.content or ""
        return {"text": content, "usage": usage}

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        response_format: dict | None = None,
        max_completion_tokens: int | None = _SENTINEL,
    ) -> AsyncGenerator[str, None]:
        full_messages = _prepend_system(system, messages)
        # include_usage adds a final SSE chunk with the same usage block as
        # non-streaming responses. Without it, streaming responses have no
        # per-request token data.
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Match the non-streaming path: when the caller asks for JSON mode,
        # forward it so the model's output is constrained. Skipping this on
        # the streaming path was the root cause of BaseAgent's ReAct loop
        # accepting prose responses and crashing in ``_parse_action_json``.
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        # Sentinel-vs-None distinction so callers can explicitly pass
        # ``max_completion_tokens=None`` to opt out of the cap (let model
        # use its own ceiling). The instance default fills in otherwise.
        effective_max = (
            self._max_completion_tokens
            if max_completion_tokens is _SENTINEL
            else max_completion_tokens
        )
        if effective_max is not None:
            request_kwargs["max_completion_tokens"] = effective_max
        raw = await self._client.chat.completions.with_raw_response.create(**request_kwargs)
        headers = _headers_dict(raw)
        stream = raw.parse()
        final_chunk = None
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                # OpenAI sends the usage on a chunk whose choices is empty.
                final_chunk = chunk
                continue
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        if final_chunk is not None:
            usage = self._build_usage(final_chunk, headers)
            self._record_usage(usage, source=source)
            self.last_usage = usage

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_usage(self, resp: Any, headers: dict[str, str] | None) -> dict:
        usage_obj = getattr(resp, "usage", None)
        # Defensive: some chunks may lack usage entirely. Skip silently.
        if usage_obj is None:
            return {}
        usage = {
            "tokens_in": getattr(usage_obj, "prompt_tokens", 0),
            "tokens_out": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
            "model": getattr(resp, "model", self._model),
        }
        # Gateway header wins if present — it's the most authoritative cost.
        cost = _read_gateway_cost(headers)
        # Otherwise let the caller compute from token counts.
        if cost is None and self._cost_fn is not None:
            try:
                cost = float(self._cost_fn(usage))
            except Exception as e:
                logger.warning("cost_fn raised: %s — skipping cost for this call", e)
                cost = None
        if cost is not None:
            usage["cost_usd"] = cost
        return usage

    def _record_usage(self, usage: dict, *, source: str | None) -> None:
        """Forward usage to the budget guard.

        Tokens are reported on every call (even when no ``cost_fn`` is wired)
        so token-based caps still fire. Cost is forwarded only when known.
        Both calls accept the per-call-site ``source`` tag so the guard's
        breakdown attributes spending to the right slot.
        """
        guard = self._budget
        if not guard:
            return
        tokens_in = int(usage.get("tokens_in") or 0)
        tokens_out = int(usage.get("tokens_out") or 0)
        if (tokens_in or tokens_out) and hasattr(guard, "add_tokens"):
            guard.add_tokens(tokens_in, tokens_out, source=source)
        cost = usage.get("cost_usd")
        if cost and cost > 0:
            guard.add_cost(cost, source=source)


# ── Module-level helpers ─────────────────────────────────────────────────────


def _lookup_openai_context_window(model: str) -> int:
    """Resolve a model id to a context window via ``_OPENAI_CONTEXT_WINDOWS``.

    Exact match wins; then longest-prefix match (so ``gpt-4o-mini-2024-…``
    resolves to ``gpt-4o-mini``'s 128K, not ``gpt-4o``'s). Unknown models
    fall back to ``_OPENAI_CONTEXT_WINDOW_FALLBACK`` plus a warning so the
    user notices and can pass ``context_window=N`` explicitly.
    """
    if model in _OPENAI_CONTEXT_WINDOWS:
        return _OPENAI_CONTEXT_WINDOWS[model]
    # Sort prefixes by length descending so more-specific entries win.
    for prefix in sorted(_OPENAI_CONTEXT_WINDOWS, key=len, reverse=True):
        if model.startswith(prefix):
            return _OPENAI_CONTEXT_WINDOWS[prefix]
    logger.warning(
        "OpenAILLM: unknown model %r — defaulting context_window to %d. "
        "Pass `context_window=N` to OpenAILLM(...) for the real value.",
        model,
        _OPENAI_CONTEXT_WINDOW_FALLBACK,
    )
    return _OPENAI_CONTEXT_WINDOW_FALLBACK


def _prepend_system(system: str | None, messages: list[dict]) -> list[dict]:
    """If a separate system prompt is provided, inject it as the first message."""
    if not system:
        return list(messages)
    return [{"role": "system", "content": system}, *messages]


def _headers_dict(raw: Any) -> dict[str, str]:
    """Normalize openai's APIResponse.headers (which is httpx.Headers-like) to a plain dict."""
    headers = getattr(raw, "headers", None)
    if headers is None:
        return {}
    try:
        return {k.lower(): v for k, v in headers.items()}
    except Exception:
        return {}


def _read_gateway_cost(headers: dict[str, str] | None) -> float | None:
    if not headers:
        return None
    for name in _GATEWAY_COST_HEADERS:
        if name in headers:
            try:
                return float(headers[name])
            except (ValueError, TypeError):
                logger.warning(
                    "gateway cost header %s present but unparseable: %r", name, headers[name]
                )
                return None
    return None
