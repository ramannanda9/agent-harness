"""
FetchImage — download a remote image and return an OpenAI-compatible image_url content block.

The returned dict slots directly into an LLM message's `content` list, which
WorkingMemory passes through as-is to the vision-capable model:

    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "auto"}}

Detail levels (OpenAI):
  "low"  — fixed 85 tokens, low-res 512×512 tile
  "high" — up to ~1700 tokens, full resolution
  "auto" — model decides (default)

Install:
    pip install -e ".[http]"   # httpx is already pulled in by the http extra

On error returns:
    {"error": "<message>"}
"""

from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB hard cap


class FetchImage:
    name = "fetch_image"

    def __init__(
        self,
        *,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = MAX_IMAGE_BYTES,
        follow_redirects: bool = True,
        detail: str = "auto",
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError('httpx not installed. Run: pip install -e ".[http]"') from e
        self._timeout = default_timeout_seconds
        self._max_bytes = max_bytes
        self._follow_redirects = follow_redirects
        self._detail = detail

    async def execute(self, url: str, detail: str | None = None) -> dict:
        """
        Fetch the image at `url` and return an image_url content block.

        Args:
            url:    HTTP/HTTPS URL of the image.
            detail: OpenAI detail level override — "auto", "low", or "high".

        Returns image_url content block on success, {"error": "..."} on failure.
        """
        import httpx

        effective_detail = detail or self._detail

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=self._follow_redirects,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except Exception as e:
            logger.warning("fetch_image failed for %s: %s", url, e)
            return {"error": f"{type(e).__name__}: {e}"}

        raw = resp.content
        if len(raw) > self._max_bytes:
            return {"error": f"image too large: {len(raw)} bytes (limit {self._max_bytes})"}

        mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        b64 = base64.b64encode(raw).decode()

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": effective_detail,
            },
        }
