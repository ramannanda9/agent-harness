"""Shared utilities for harness, orchestrator, and memory packages."""

from __future__ import annotations

import json
from typing import Any


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
