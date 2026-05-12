"""
OpenAI LLM adapter.

Implements the harness's LLM client contract:
  - async def complete(system, messages, **kwargs) -> dict
  - async def stream_complete(system, messages) -> AsyncGenerator[str, None]

The harness uses `system=None` for the agent ReAct path (the system prompt
sits inside `messages` as the first message). For orchestrator/memory paths
(planning, synthesis, extraction, summarization) `system` is a string. This
adapter prepends it as a "system" role message when provided.

Install:
    pip install -e ".[openai]"

Usage:
    from harness.llm.openai import OpenAILLM
    llm = OpenAILLM(model="gpt-4o-mini")        # reads OPENAI_API_KEY from env
    runtime = AgentRuntime(..., llm=llm)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)


class OpenAILLM:
    def __init__(
        self,
        *,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,  # falls back to OPENAI_API_KEY env
        request_timeout_seconds: float = 60.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError('openai package not installed. Run: pip install -e ".[openai]"') from e

        self._client = AsyncOpenAI(api_key=api_key, timeout=request_timeout_seconds)
        self._model = model

    # ── Non-streaming ─────────────────────────────────────────────────────────

    async def complete(
        self,
        system: str | None,
        messages: list[dict],
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

        resp = await self._client.chat.completions.create(**request)
        content = resp.choices[0].message.content or ""
        return {"text": content}

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        full_messages = _prepend_system(system, messages)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _prepend_system(system: str | None, messages: list[dict]) -> list[dict]:
    """If a separate system prompt is provided, inject it as the first message."""
    if not system:
        return list(messages)
    return [{"role": "system", "content": system}, *messages]
