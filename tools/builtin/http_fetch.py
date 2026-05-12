"""
HTTPFetch — minimal read-only GET tool.

Intentionally boring and safe. For production / authed / retry-heavy use,
write your own tool against httpx or aiohttp. This one exists so the harness
has a useful out-of-box tool for examples and smoke tests.

Body is capped at `max_bytes` (default 64 KiB) so the LLM doesn't accidentally
ingest a 50 MB response. Truncation is signaled in the return value so the
agent can reason about it.

Install:
    pip install -e ".[http]"

Returned dict shape:
    {
        "status":       int,                # HTTP status code
        "content_type": str,                # response Content-Type header (or "")
        "body":         str,                # response text, possibly truncated
        "truncated":    bool,                # True if body was capped
        "url":          str,                # final URL after redirects
    }

On failure (network error, timeout, etc) returns:
    {"error": "<message>"}
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 64 * 1024
DEFAULT_TIMEOUT_SECONDS = 10.0


class HTTPFetch:
    name = "http_fetch"

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        follow_redirects: bool = True,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "httpx not installed. Run: pip install -e \".[http]\""
            ) from e
        self._max_bytes = max_bytes
        self._timeout = default_timeout_seconds
        self._follow_redirects = follow_redirects

    async def execute(
        self,
        url: str,
        timeout_seconds: float | None = None,
    ) -> dict:
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds or self._timeout,
                follow_redirects=self._follow_redirects,
            ) as client:
                resp = await client.get(url)
        except Exception as e:
            logger.warning("http_fetch failed for %s: %s", url, e)
            return {"error": f"{type(e).__name__}: {e}"}

        text = resp.text
        truncated = len(text) > self._max_bytes
        if truncated:
            text = text[: self._max_bytes]

        return {
            "status": resp.status_code,
            "content_type": resp.headers.get("content-type", ""),
            "body": text,
            "truncated": truncated,
            "url": str(resp.url),
        }
