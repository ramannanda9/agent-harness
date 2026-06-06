"""Direct OpenAI Codex backend adapter.

This is the Pi-style route: OAuth credentials are read from a Pi-shaped auth
file and requests go directly to the Codex backend:

    https://chatgpt.com/backend-api/codex/responses

The normal OpenAI API-key adapter remains `harness.llm.openai.OpenAILLM`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from harness.llm._streaming import aiter_sse_events, format_streaming_error, read_error_body
from harness.llm.auth import AuthFileOAuthProvider, OAuthCredential, OpenAICodexOAuthClient


class OpenAICodexLLM:
    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        auth_file: str | Path | None = None,
        credential_provider: AuthFileOAuthProvider | None = None,
        base_url: str = "https://chatgpt.com/backend-api",
        request_timeout_seconds: float = 120.0,
        http_client: Any | None = None,
        codex_originator: str = "agent-harness",
    ) -> None:
        if credential_provider is None:
            if auth_file is None:
                auth_file = Path("~/.agent-harness/auth/auth.json").expanduser()
            oauth = OpenAICodexOAuthClient()
            credential_provider = AuthFileOAuthProvider(
                auth_file,
                provider="openai-codex",
                refresher=oauth.refresh,
            )
        self._credentials = credential_provider
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout_seconds
        self._client = http_client
        self._owns_client = http_client is None
        self._codex_originator = codex_originator
        self._budget: Any = None
        self.last_usage: dict | None = None

    def set_budget(self, guard: Any) -> None:
        """Inject a BudgetGuard so token caps fire on subscription-auth runs.

        Cost stays 0 (no pricing schedule available for the subscription
        tier), but ``add_tokens`` still lands so ``max_input_tokens`` /
        ``max_output_tokens`` are enforced.
        """
        self._budget = guard

    async def complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Collect the streaming response into a single text + usage dict.

        Internally consumes the same SSE stream as `stream_complete` —
        there is no separate non-streaming code path. The Codex backend
        only returns SSE; we just buffer the deltas before returning.
        """
        extra = dict(kwargs)
        extra.pop("max_output_tokens", None)
        text_parts: list[str] = []
        async for delta in self._iter_stream(system, messages, extra=extra, source=source):
            text_parts.append(delta)
        text = "".join(text_parts)
        usage = self.last_usage or {}
        if not text:
            raise RuntimeError("Codex SSE response did not contain output text")
        return {"text": text, "usage": usage}

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Yield each `response.output_text.delta` token as it arrives.

        ``kwargs`` accepts OpenAI-style hints like ``response_format`` so
        the same ReAct-driving caller works against this adapter and the
        public ``OpenAILLM``; Codex's responses backend wires JSON output
        differently and the kwarg is intentionally ignored. Codex currently
        rejects ``max_output_tokens``, so that public Responses API option is
        filtered out when shared harness code passes it through.
        """
        extra = dict(kwargs)
        extra.pop("max_output_tokens", None)
        async for delta in self._iter_stream(system, messages, extra=extra, source=source):
            yield delta

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    # ── Streaming core ────────────────────────────────────────────────────────

    async def _iter_stream(
        self,
        system: str | None,
        messages: list[dict],
        *,
        extra: dict[str, Any],
        source: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Single source of truth: open an SSE stream, yield deltas, set
        `self.last_usage` from the final event. Auth refresh on 401/403
        happens before any delta is yielded so we never retry mid-stream.
        """
        payload = _build_payload(
            model=self._model,
            system=system,
            messages=messages,
            extra=extra,
        )
        url = f"{self._base_url}/codex/responses"

        for attempt in range(2):
            cred = await self._credentials.get_credential(force_refresh=(attempt == 1))
            client = await self._get_client()
            headers = _build_headers(cred, originator=self._codex_originator)

            async with client.stream("POST", url, headers=headers, json=payload) as response:
                status = getattr(response, "status_code", 200)
                if status in (401, 403) and attempt == 0:
                    continue  # closes stream, retries with fresh creds
                if status >= 400:
                    body = await read_error_body(response)
                    raise RuntimeError(format_streaming_error(status, body, provider="Codex"))

                final_payload: dict[str, Any] | None = None
                yielded_any = False
                async for event_type, data in aiter_sse_events(response):
                    if not data or data == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if event_type == "response.output_text.delta":
                        delta = parsed.get("delta") if isinstance(parsed, dict) else None
                        if isinstance(delta, str) and delta:
                            yielded_any = True
                            yield delta
                    elif event_type in {"response.completed", "response.done"}:
                        if isinstance(parsed, dict):
                            response_payload = (
                                parsed.get("response")
                                if isinstance(parsed.get("response"), dict)
                                else parsed
                            )
                            if isinstance(response_payload, dict):
                                final_payload = response_payload

                usage: dict = {}
                if final_payload is not None:
                    usage = _normalize_usage(final_payload.get("usage"))
                    if not yielded_any:
                        # Some backends only emit the full text in the final payload.
                        try:
                            text = _extract_output_text(final_payload)
                        except RuntimeError:
                            text = ""
                        if text:
                            yield text
                self.last_usage = usage
                self._record_usage(usage, source=source)
                return

        raise RuntimeError("Codex authentication failed after refresh")

    def _record_usage(self, usage: dict, *, source: str | None) -> None:
        """Report token totals to the budget guard.

        Codex backend reports `tokens_in` / `tokens_out` directly; cached
        input tokens are tracked separately but billed at the same wall-clock
        rate from the user's perspective, so they count toward
        ``max_input_tokens`` too. No cost is reported.
        """
        guard = self._budget
        if not guard or not hasattr(guard, "add_tokens"):
            return
        tokens_in = int(usage.get("tokens_in") or 0) + int(usage.get("cached_input_tokens") or 0)
        tokens_out = int(usage.get("tokens_out") or 0)
        if tokens_in or tokens_out:
            guard.add_tokens(tokens_in, tokens_out, source=source)

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    'httpx package not installed. Run: pip install -e ".[http]"'
                ) from e
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client


def _build_headers(cred: OAuthCredential, *, originator: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {cred.access}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "agent-harness",
        "originator": originator,
        "OpenAI-Beta": "responses=v1",
    }
    if cred.account_id:
        headers["chatgpt-account-id"] = cred.account_id
    return headers


def _build_payload(
    *,
    model: str,
    system: str | None,
    messages: list[dict],
    extra: dict[str, Any],
) -> dict[str, Any]:
    input_items = []
    instructions = system or "You are a helpful assistant."
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            text = _content_to_text(message.get("content", ""))
            if text:
                instructions = f"{instructions}\n\n{text}" if instructions else text
            continue
        input_items.append(
            {
                "type": "message",
                "role": role,
                "content": [
                    {
                        "type": "input_text" if role != "assistant" else "output_text",
                        "text": _content_to_text(message.get("content", "")),
                    }
                ],
            }
        )
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "tools": [],
        "tool_choice": "none",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
        "include": [],
    }
    for key in ("reasoning", "temperature", "service_tier", "text"):
        if key in extra:
            payload[key] = extra[key]
    return payload


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
            elif isinstance(content, str):
                parts.append(content)
        if parts:
            return "".join(parts)

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]

    raise RuntimeError("Codex response did not contain output text")


def _normalize_usage(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {}
    tokens_in = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
    tokens_out = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": int(raw.get("total_tokens") or tokens_in + tokens_out),
        "cached_input_tokens": int(raw.get("cached_input_tokens") or 0),
        "reasoning_output_tokens": int(raw.get("reasoning_output_tokens") or 0),
        "provider": "openai-codex",
    }
