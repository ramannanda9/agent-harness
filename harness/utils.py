"""Shared utilities for harness, orchestrator, and memory packages."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Coroutine
from typing import Any

_log = logging.getLogger(__name__)


def fire(coro: Coroutine) -> asyncio.Task:
    """Schedule coro as a background task; log but don't propagate exceptions.

    Use for best-effort writes (memory, metrics) that must not block the
    critical path. The returned Task can be ignored — errors are logged.
    """
    task = asyncio.create_task(coro)

    def _on_done(t: asyncio.Task) -> None:
        if not t.cancelled() and (exc := t.exception()):
            _log.warning("Background task failed: %s", exc)

    task.add_done_callback(_on_done)
    return task


def parse_llm_json(response: Any) -> dict:
    """Unwrap an LLM adapter response into a plain dict.

    LLM adapters may return:
      - a dict with a "text" key containing a JSON string
      - a raw JSON string
      - a dict already (json_object mode with some adapters)

    Raises json.JSONDecodeError if the content is not valid JSON.
    """
    if isinstance(response, dict) and "text" in response:
        return json.loads(response["text"])
    if isinstance(response, str):
        return json.loads(response)
    if isinstance(response, dict):
        return response
    return {}
