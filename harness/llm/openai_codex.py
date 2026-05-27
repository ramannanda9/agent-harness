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
        self.last_usage: dict | None = None

    async def complete(
        self,
        system: str | None,
        messages: list[dict],
        **kwargs: Any,
    ) -> dict:
        payload = _build_payload(
            model=self._model,
            system=system,
            messages=messages,
            extra=kwargs,
        )
        response = await self._post_with_auth_retry(payload)
        text, usage = _parse_sse_response(response)
        self.last_usage = usage
        return {"text": text, "usage": usage}

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        payload = _build_payload(
            model=self._model,
            system=system,
            messages=messages,
            extra={},
        )
        response = await self._post_with_auth_retry(payload)
        text, usage = _parse_sse_response(response)
        self.last_usage = usage
        yield text

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _post_with_auth_retry(self, payload: dict[str, Any]) -> Any:
        cred = await self._credentials.get_credential()
        response = await self._post(payload, cred)
        if getattr(response, "status_code", None) in (401, 403):
            cred = await self._credentials.get_credential(force_refresh=True)
            response = await self._post(payload, cred)
        if getattr(response, "status_code", 200) >= 400:
            raise RuntimeError(_format_error_response(response))
        return response

    async def _post(self, payload: dict[str, Any], cred: OAuthCredential) -> Any:
        client = await self._get_client()
        headers = _build_headers(cred, originator=self._codex_originator)
        return await client.post(
            f"{self._base_url}/codex/responses",
            headers=headers,
            json=payload,
        )

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
    for key in ("reasoning", "max_output_tokens", "temperature", "service_tier", "text"):
        if key in extra:
            payload[key] = extra[key]
    return payload


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def _parse_response(payload: dict[str, Any]) -> tuple[str, dict]:
    text = _extract_output_text(payload)
    usage = _normalize_usage(payload.get("usage"))
    return text, usage


def _parse_sse_response(response: Any) -> tuple[str, dict]:
    text_parts: list[str] = []
    usage: dict = {}
    final_payload: dict[str, Any] | None = None
    for event in _iter_sse_events(_response_text(response)):
        event_type = event.get("event")
        data = event.get("data")
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
        elif event_type in {"response.completed", "response.done"}:
            response_payload = (
                payload.get("response") if isinstance(payload.get("response"), dict) else payload
            )
            final_payload = response_payload if isinstance(response_payload, dict) else None

    if final_payload is not None:
        usage = _normalize_usage(final_payload.get("usage"))
        if not text_parts:
            try:
                return _extract_output_text(final_payload), usage
            except RuntimeError:
                pass
    if text_parts:
        return "".join(text_parts), usage
    raise RuntimeError("Codex SSE response did not contain output text")


def _iter_sse_events(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current_event = "message"
    data_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                events.append({"event": current_event, "data": "\n".join(data_lines)})
                current_event = "message"
                data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    if data_lines:
        events.append({"event": current_event, "data": "\n".join(data_lines)})
    return events


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode(errors="replace")
    raise RuntimeError("Codex response did not include SSE text")


def _format_error_response(response: Any) -> str:
    status = getattr(response, "status_code", "unknown")
    try:
        data = response.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        detail = data.get("detail")
        error = data.get("error")
        if isinstance(detail, str):
            return f"Codex backend returned {status}: {detail}"
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or error
            return f"Codex backend returned {status}: {message}"
        if error:
            return f"Codex backend returned {status}: {error}"
    text = (
        _response_text(response)
        if hasattr(response, "text") or hasattr(response, "content")
        else ""
    )
    if text.strip():
        return f"Codex backend returned {status}: {text.strip()[:500]}"
    return f"Codex backend returned HTTP {status}"


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
