"""``RoutingLLM`` — dispatch each LLM call to a different adapter by a selector.

**For agent-harness's own call sites, prefer per-call-site injection** —
``AgentRuntime`` exposes ``classifier_llm`` / ``router_llm`` and
``Orchestrator`` exposes ``planner_llm`` / ``synthesizer_llm``. That's the
production-style pattern: each call site is hard-wired to a model chosen
for that workload's cost / quality / latency budget, no runtime guessing.

``RoutingLLM`` is the **bring-your-own-selector primitive** for cases
where per-call-site injection isn't enough:

  - You're wrapping an existing harness instance you can't restructure.
  - You're routing based on **capability** (does this query need
    vision / function calling / >200K context?) — that's a real
    production pattern, but the metadata is provider-specific so the
    selector has to be yours.
  - You're doing **learned routing** (RouteLLM-style classifier) where
    the selector is a small ML model.
  - You're doing **cascade routing** (cheap-then-escalate-on-low-confidence)
    via a custom selector that inspects prior responses.

Wire it up with your own selector callable that returns a key from the
``routes`` dict::

    from harness.llm.routing import RoutingLLM

    def my_capability_selector(system, messages):
        # Inspect the call's requirements and pick the cheapest viable model.
        if _needs_vision(messages):
            return "vision"
        if _estimated_tokens(system, messages) > 100_000:
            return "long_context"
        return "default"

    llm = RoutingLLM(
        routes={
            "default":      OpenAILLM(model="gpt-4o-mini"),
            "vision":       OpenAILLM(model="gpt-4o"),
            "long_context": AnthropicLLM(model="claude-sonnet-4-6"),
        },
        selector=my_capability_selector,
        default_route="default",
    )

The harness does **not** ship default selectors. Naive selectors
(keyword matching, fixed token thresholds) misroute in subtle ways and
encourage the wrong mental model. If you find yourself reaching for one,
the per-call-site injection path on ``AgentRuntime`` / ``Orchestrator``
is almost certainly what you actually want.

``last_route`` exposes the key of the route that handled the most recent
call — handy for logging and tests.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)


Selector = Callable[[str | None, list[dict]], str]


class RoutingLLM:
    def __init__(
        self,
        routes: Mapping[str, Any],
        *,
        selector: Selector,
        default_route: str,
    ) -> None:
        if not routes:
            raise ValueError("RoutingLLM requires at least one route")
        if default_route not in routes:
            raise ValueError(f"default_route {default_route!r} is not in routes")
        self._routes = dict(routes)
        self._selector = selector
        self._default_route = default_route
        self.last_route: str = default_route
        self.last_usage: dict | None = None

    def set_budget(self, guard: Any) -> None:
        """Forward the budget guard to every routed LLM."""
        for llm in self._routes.values():
            if hasattr(llm, "set_budget"):
                llm.set_budget(guard)

    def _pick(self, system: str | None, messages: list[dict]) -> tuple[str, Any]:
        try:
            key = self._selector(system, messages)
        except Exception as e:  # noqa: BLE001 — fall back gracefully
            logger.warning("RoutingLLM selector raised %s — using default route", e)
            key = self._default_route
        if key not in self._routes:
            logger.warning(
                "RoutingLLM selector returned unknown key %r — using default route %r",
                key,
                self._default_route,
            )
            key = self._default_route
        return key, self._routes[key]

    # ── Non-streaming ────────────────────────────────────────────────────────

    async def complete(
        self,
        system: str | None,
        messages: list[dict],
        **kwargs: Any,
    ) -> dict:
        key, llm = self._pick(system, messages)
        self.last_route = key
        result = await llm.complete(system, messages, **kwargs)
        self.last_usage = getattr(llm, "last_usage", None)
        return result

    # ── Streaming ────────────────────────────────────────────────────────────

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        key, llm = self._pick(system, messages)
        self.last_route = key
        if not hasattr(llm, "stream_complete"):
            # Fall back to non-streaming for routes that don't implement it.
            result = await llm.complete(system, messages)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            if text:
                yield text
            self.last_usage = getattr(llm, "last_usage", None)
            return
        async for chunk in llm.stream_complete(system, messages):
            yield chunk
        self.last_usage = getattr(llm, "last_usage", None)
