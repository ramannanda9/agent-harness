"""``RoutingLLM`` — dispatch each LLM call to a different adapter by a selector.

The classic use case is **cost shaping**: route the *short, structured,
low-stakes* calls to a cheap model, keep the *reasoning-heavy* calls on a
frontier model. In this harness the natural split is:

  Cheap-appropriate (short prompts, small/enum outputs):
    - the classifier (``simple`` vs ``complex`` dispatch decision)
    - the router (pick one of N agents)
    - memory summarisation (mechanical compaction of older context)

  Premium-required (high-stakes reasoning, multi-turn outputs):
    - the planner (decomposes the goal into a DAG — bad plan, run wasted)
    - the replanner (reasons about why the previous plan failed)
    - the per-agent ReAct loop (the actual work)
    - the synthesiser when it has to reconcile conflicting evidence

Wire it up by giving each adapter a key, then supplying a ``selector``
function that returns the key for the current call::

    from harness.llm.openai import OpenAILLM
    from harness.llm.anthropic import AnthropicLLM
    from harness.llm.routing import RoutingLLM, by_system_keyword

    llm = RoutingLLM(
        routes={
            "cheap":   OpenAILLM(model="gpt-4o-mini"),
            "default": AnthropicLLM(model="claude-sonnet-4-6"),
        },
        selector=by_system_keyword(
            # Match phrases from the orchestrator's own system prompts —
            # "classifier" and "routing agent" appear verbatim.
            {"classifier": "cheap", "routing agent": "cheap"},
            default="default",
        ),
        default_route="default",
    )

The selector receives ``(system, messages)`` and returns a key from the
``routes`` dict. The shipped selectors are::

    by_system_keyword({"classifier": "cheap", ...}, default="default")
        Match substrings in the system prompt.

    by_token_count(threshold=1500, small="cheap", large="default")
        Cheap model for short contexts (good for the classifier and router
        which see only the goal + agent descriptions); bigger model once
        the working memory grows.

    by_role({"classifier": "cheap", ...}, default="default")
        Match against a ``call_role`` field on the last message — useful
        when your own call sites tag their purpose explicitly.

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

    def _pick(self, system: str | None, messages: list[dict], **kwargs: Any) -> tuple[str, Any]:
        # The selector may inspect kwargs (e.g. call_role) by reading through
        # the call site, but most selectors only care about system/messages.
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
        # call_role is consumed by selectors that need it; never forwarded to
        # the underlying adapter (would confuse SDKs that validate kwargs).
        if "call_role" in kwargs:
            kwargs.pop("call_role")
        key, llm = self._pick(system, messages, **kwargs)
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


# ── Selectors ────────────────────────────────────────────────────────────────


def by_system_keyword(
    mapping: Mapping[str, str],
    *,
    default: str,
) -> Selector:
    """Route by case-insensitive substring match against the system prompt.

    Order of the mapping matters — first matching keyword wins. Provide an
    ``OrderedDict`` if your Python version doesn't preserve insertion order
    (Python 3.7+ regular dicts do).
    """
    items = list(mapping.items())

    def select(system: str | None, _messages: list[dict]) -> str:
        if not system:
            return default
        haystack = system.lower()
        for needle, route in items:
            if needle.lower() in haystack:
                return route
        return default

    return select


def by_token_count(
    *,
    threshold: int,
    small: str,
    large: str,
    counter: Callable[[str], int] | None = None,
) -> Selector:
    """Route to ``small`` when total characters fit under ``threshold`` * 4,
    else route to ``large``.

    Pass a ``counter`` callable for accurate token counting (e.g. tiktoken);
    the default char-divided-by-4 heuristic is good enough for routing.
    """

    def _chars(s: str) -> int:
        return len(s)

    count = counter or (lambda s: _chars(s) // 4)

    def select(system: str | None, messages: list[dict]) -> str:
        total = count(system or "")
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += count(str(block.get("text") or ""))
        return small if total < threshold else large

    return select


def by_role(
    mapping: Mapping[str, str],
    *,
    default: str,
) -> Selector:
    """Route by an explicit ``call_role`` kwarg passed by the caller.

    The harness doesn't yet thread a role kwarg through every LLM call site,
    so this selector is most useful when you build your own call sites that
    explicitly tag their purpose. The other selectors work without changes
    to the harness.
    """

    def select(_system: str | None, messages: list[dict]) -> str:
        # call_role propagates via the message envelope when callers stash it
        # there; otherwise we fall back to the default. Inspect the last
        # message's metadata for a ``call_role`` field.
        for m in reversed(messages):
            role = m.get("call_role") if isinstance(m, dict) else None
            if isinstance(role, str) and role in mapping:
                return mapping[role]
        return default

    return select
