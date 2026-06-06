"""Direct Claude Code-style Anthropic OAuth adapter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from harness.llm._streaming import aiter_sse_events, format_streaming_error, read_error_body
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

# Memoized installed CLI version probe — `claude --version` is slow enough that
# we don't want to re-run it for every request. Env overrides bypass the cache.
_cached_cli_version: str | None = None
_cli_version_probed = False


class ClaudeCodeLLM:
    def __init__(
        self,
        *,
        model: str | None = None,
        auth_file: str | Path | None = None,
        credential_provider: AuthFileOAuthProvider | None = None,
        base_url: str = "https://api.anthropic.com",
        request_timeout_seconds: float = 120.0,
        # Matches AnthropicLLM's bumped default. The original 1024 cap
        # clipped JSON-mode responses in long ReAct loops; 4096 keeps
        # headroom without blowing through per-call budgets.
        max_tokens: int = 4096,
        http_client: Any | None = None,
        user_agent: str | None = None,
        betas: str = CLAUDE_CODE_BETAS,
        prompt_caching: bool = True,
        # Explicit context window override; falls back to the Anthropic
        # lookup table since Claude Code is Anthropic underneath.
        context_window: int | None = None,
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
        self._prompt_caching = prompt_caching
        # Reuse Anthropic's table — Claude Code is Anthropic underneath.
        from harness.llm.anthropic import (  # noqa: PLC0415
            _ANTHROPIC_TOKEN_BUDGET_SAFETY,
            _lookup_anthropic_context_window,
        )

        self._context_window = context_window or _lookup_anthropic_context_window(self._model)
        self._token_budget_safety = _ANTHROPIC_TOKEN_BUDGET_SAFETY
        self._budget: Any = None
        self.last_usage: dict | None = None

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def input_token_budget(self) -> int:
        """Tokens available for the prompt; Anthropic semantics — context is
        input-only, output cap is separate."""
        return max(1024, self._context_window - self._token_budget_safety)

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
        there is no separate non-streaming code path. We pass
        `stream=true` and accumulate the deltas.
        """
        max_tokens = int(kwargs.pop("max_tokens", self._max_tokens))
        parts: list[str] = []
        async for delta in self._iter_stream(
            system, messages, max_tokens=max_tokens, extra=kwargs, source=source
        ):
            parts.append(delta)
        text = "".join(parts)
        if not text:
            raise RuntimeError("Claude Code response did not contain text content")
        return {"text": text, "usage": self.last_usage or {}}

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        **_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        # ``_kwargs`` swallows OpenAI-style hints like ``response_format``;
        # Claude Code (Anthropic underneath) doesn't expose an equivalent.
        async for delta in self._iter_stream(
            system, messages, max_tokens=self._max_tokens, extra={}, source=source
        ):
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
        max_tokens: int,
        extra: dict[str, Any],
        source: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Single source of truth: open Anthropic SSE stream, yield text
        deltas, populate `self.last_usage`. Auth refresh on 401/403
        happens before any delta is yielded so we never retry mid-stream.
        """
        payload = _build_payload(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            extra=extra,
            prompt_caching=self._prompt_caching,
        )
        payload["stream"] = True
        url = f"{self._base_url}/v1/messages"

        for attempt in range(2):
            cred = await self._credentials.get_credential(force_refresh=(attempt == 1))
            client = await self._get_client()
            headers = _build_headers(cred, user_agent=self._user_agent, betas=self._betas)

            async with client.stream("POST", url, headers=headers, json=payload) as response:
                status = getattr(response, "status_code", 200)
                if status in (401, 403) and attempt == 0:
                    continue  # closes stream, retries with fresh creds
                if status >= 400:
                    body = await read_error_body(response)
                    raise RuntimeError(format_streaming_error(status, body, provider="Claude Code"))

                tokens_in = 0
                tokens_out = 0
                cache_read_tokens = 0
                cache_creation_tokens = 0
                async for _event_type, data in aiter_sse_events(response):
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    otype = obj.get("type")
                    if otype == "content_block_delta":
                        delta = obj.get("delta")
                        if isinstance(delta, dict) and delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if isinstance(text, str) and text:
                                yield text
                    elif otype == "message_start":
                        msg_usage = (obj.get("message") or {}).get("usage") or {}
                        tokens_in = int(msg_usage.get("input_tokens") or 0)
                        cache_read_tokens = int(msg_usage.get("cache_read_input_tokens") or 0)
                        cache_creation_tokens = int(
                            msg_usage.get("cache_creation_input_tokens") or 0
                        )
                    elif otype == "message_delta":
                        delta_usage = obj.get("usage") or {}
                        tokens_out = int(delta_usage.get("output_tokens") or 0)

                self.last_usage = {
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "total_tokens": tokens_in + tokens_out,
                    "provider": "claude-code",
                }
                self._record_usage(self.last_usage, source=source)
                return

        raise RuntimeError("Claude Code authentication failed after refresh")

    def _record_usage(self, usage: dict, *, source: str | None) -> None:
        """Report token totals to the budget guard.

        Tokens budgeted = total input that hit the wire (non-cached +
        cache-creation + cache-read) plus output tokens — so ``max_input_tokens``
        / ``max_output_tokens`` reflect real consumption regardless of cache
        hit rate. No cost is reported (subscription auth, no pricing).
        """
        guard = self._budget
        if not guard or not hasattr(guard, "add_tokens"):
            return
        tokens_in = (
            int(usage.get("tokens_in") or 0)
            + int(usage.get("cache_read_tokens") or 0)
            + int(usage.get("cache_creation_tokens") or 0)
        )
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
    return f"claude-cli/{_resolve_cc_version()} (external, cli)"


def _resolve_cc_version() -> str:
    """Resolve the Claude Code CLI version, used in User-Agent + billing header.

    Order:
      1. `CLAUDE_CODE_VERSION` env (callers can pin a known-good version).
      2. `claude --version` from the installed CLI (cached for the process —
         the subprocess probe is too slow to repeat per request).
      3. `"unknown"` fallback so the header still has a valid shape.
    """
    env_version = os.environ.get("CLAUDE_CODE_VERSION")
    if env_version:
        return env_version
    global _cached_cli_version, _cli_version_probed
    if not _cli_version_probed:
        _cached_cli_version = _installed_claude_version()
        _cli_version_probed = True
    return _cached_cli_version or "unknown"


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
    prompt_caching: bool = True,
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
    built_messages = [_message_payload(message) for message in input_messages]
    if prompt_caching:
        _apply_last_user_cache_control(built_messages)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _system_blocks(instructions, prompt_caching=prompt_caching),
        "messages": built_messages,
    }
    for key in ("temperature", "top_p", "top_k", "stop_sequences", "thinking"):
        if key in extra:
            payload[key] = extra[key]
    return payload


def _system_blocks(system: str | None, *, prompt_caching: bool = True) -> list[dict[str, Any]]:
    cc_version = _resolve_cc_version()
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"x-anthropic-billing-header: cc_version={cc_version}; "
                "cc_entrypoint=cli; cch=00000;"
            ),
        },
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
    ]
    if system:
        block: dict[str, Any] = {"type": "text", "text": system}
        if prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def _apply_last_user_cache_control(messages: list[dict]) -> None:
    """Add cache_control to the last user message's content block (string only).

    This marks the current task/goal as cacheable so repeated ReAct steps
    that share the same leading conversation prefix benefit from the cache.
    Only mutates messages whose last user-role entry has a plain-string
    content block (skips multimodal / already-list content).
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            break
        # content is already a list of blocks from _message_payload
        if len(content) == 1 and content[0].get("type") == "text":
            content[0]["cache_control"] = {"type": "ephemeral"}
        break


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
