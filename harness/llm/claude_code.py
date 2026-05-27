"""Direct Claude Code-style Anthropic OAuth adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from harness.llm.auth import (
    AnthropicClaudeCodeOAuthClient,
    AuthFileOAuthProvider,
    OAuthCredential,
)

CLAUDE_CODE_BETAS = os.environ.get(
    "CLAUDE_CODE_BETAS",
    "claude-code-20250219,oauth-2025-04-20",
)
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


class ClaudeCodeLLM:
    def __init__(
        self,
        *,
        model: str | None = None,
        auth_file: str | Path | None = None,
        credential_provider: AuthFileOAuthProvider | None = None,
        base_url: str = "https://api.anthropic.com",
        request_timeout_seconds: float = 120.0,
        max_tokens: int = 1024,
        http_client: Any | None = None,
        user_agent: str | None = None,
        betas: str = CLAUDE_CODE_BETAS,
    ) -> None:
        if credential_provider is None:
            if auth_file is None:
                auth_file = Path("~/.agent-harness/auth/auth.json").expanduser()
            oauth = AnthropicClaudeCodeOAuthClient()
            credential_provider = AuthFileOAuthProvider(
                auth_file,
                provider="claude-code",
                refresher=oauth.refresh,
            )
        self._credentials = credential_provider
        self._model = (
            model
            or os.environ.get("CLAUDE_CODE_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
            or "claude-sonnet-4-6"
        )
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout_seconds
        self._max_tokens = max_tokens
        self._client = http_client
        self._owns_client = http_client is None
        self._user_agent = user_agent or _default_user_agent()
        self._betas = betas
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
            max_tokens=int(kwargs.pop("max_tokens", self._max_tokens)),
            extra=kwargs,
        )
        response = await self._post_with_auth_retry(payload)
        text, usage = _parse_response(response)
        self.last_usage = usage
        return {"text": text, "usage": usage}

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        result = await self.complete(system, messages)
        yield result["text"]

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _post_with_auth_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        cred = await self._credentials.get_credential()
        response = await self._post(payload, cred)
        if getattr(response, "status_code", None) in (401, 403):
            cred = await self._credentials.get_credential(force_refresh=True)
            response = await self._post(payload, cred)
        if getattr(response, "status_code", 200) >= 400:
            raise RuntimeError(_format_error_response(response))
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Claude Code response was not a JSON object")
        return data

    async def _post(self, payload: dict[str, Any], cred: OAuthCredential) -> Any:
        client = await self._get_client()
        return await client.post(
            f"{self._base_url}/v1/messages",
            headers=_build_headers(cred, user_agent=self._user_agent, betas=self._betas),
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


def _build_headers(cred: OAuthCredential, *, user_agent: str, betas: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {cred.access}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": user_agent,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": betas,
        "x-app": "cli",
    }


def _default_user_agent() -> str:
    configured = os.environ.get("CLAUDE_CODE_USER_AGENT")
    if configured:
        return configured
    version = os.environ.get("CLAUDE_CODE_VERSION")
    if not version:
        version = _installed_claude_version()
    if version:
        return f"claude-cli/{version} (external, cli)"
    return "claude-cli/unknown (external, cli)"


def _installed_claude_version() -> str | None:
    if not shutil.which("claude"):
        return None
    try:
        proc = subprocess.run(
            ["claude", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    version = proc.stdout.strip().split(" ", 1)[0]
    return version or None


def _build_payload(
    *,
    model: str,
    system: str | None,
    messages: list[dict],
    max_tokens: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    instructions = system or ""
    input_messages: list[dict] = []
    for message in messages:
        if message.get("role") == "system":
            text = _content_to_text(message.get("content", ""))
            if text:
                instructions = f"{instructions}\n\n{text}" if instructions else text
            continue
        input_messages.append(message)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _system_blocks(instructions),
        "messages": [_message_payload(message) for message in input_messages],
    }
    for key in ("temperature", "top_p", "top_k", "stop_sequences", "thinking"):
        if key in extra:
            payload[key] = extra[key]
    return payload


def _system_blocks(system: str | None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "x-anthropic-billing-header: cc_version=2.1.97; cc_entrypoint=cli; cch=00000;",
        },
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
    ]
    if system:
        blocks.append(
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        )
    return blocks


def _message_payload(message: dict) -> dict[str, Any]:
    role = message.get("role", "user")
    if role not in {"user", "assistant"}:
        role = "user"
    return {
        "role": role,
        "content": [{"type": "text", "text": _content_to_text(message.get("content", ""))}],
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _parse_response(payload: dict[str, Any]) -> tuple[str, dict]:
    content = payload.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    usage = _normalize_usage(payload.get("usage"))
    if parts:
        return "".join(parts), usage
    raise RuntimeError("Claude Code response did not contain text content")


def _normalize_usage(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {}
    tokens_in = int(raw.get("input_tokens") or 0)
    tokens_out = int(raw.get("output_tokens") or 0)
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "provider": "claude-code",
    }


def _format_error_response(response: Any) -> str:
    status = getattr(response, "status_code", "unknown")
    try:
        data = response.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or error
            return f"Claude Code backend returned {status}: {message}"
        if error:
            return f"Claude Code backend returned {status}: {error}"
    text = getattr(response, "text", "")
    if isinstance(text, str) and text.strip():
        return f"Claude Code backend returned {status}: {text.strip()[:500]}"
    return f"Claude Code backend returned HTTP {status}"
