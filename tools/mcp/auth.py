"""Authentication helpers for MCP server connections."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness.llm.auth import AuthFileOAuthProvider


@dataclass(frozen=True)
class MCPAuth:
    """Resolved auth material for one MCP server transport."""

    headers: Mapping[str, str] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)


class MCPAuthProvider(Protocol):
    async def get_auth(self) -> MCPAuth:
        """Return headers/env for a single MCP server connection."""


class StaticMCPAuth:
    """Static MCP auth headers/env for simple tokens and private deployments."""

    def __init__(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._auth = MCPAuth(
            headers={k: v for k, v in dict(headers or {}).items() if v},
            env={k: v for k, v in dict(env or {}).items() if v},
        )

    async def get_auth(self) -> MCPAuth:
        return self._auth


class BearerMCPAuth:
    """Static bearer token auth for remote MCP servers."""

    def __init__(self, token: str, *, header_name: str = "Authorization") -> None:
        token = token.strip()
        if not token:
            raise ValueError("token must be non-empty")
        self._token = token
        self._header_name = header_name

    async def get_auth(self) -> MCPAuth:
        return MCPAuth(headers={self._header_name: f"Bearer {self._token}"})


class OAuthMCPAuth:
    """Bearer auth backed by an auth.json OAuth provider entry."""

    def __init__(
        self,
        credential_provider: AuthFileOAuthProvider,
        *,
        header_name: str = "Authorization",
    ) -> None:
        self._credentials = credential_provider
        self._header_name = header_name

    @classmethod
    def from_auth_file(
        cls,
        path: str | os.PathLike[str],
        *,
        provider: str,
        header_name: str = "Authorization",
        require_private_permissions: bool = True,
    ) -> OAuthMCPAuth:
        return cls(
            AuthFileOAuthProvider(
                path,
                provider=provider,
                require_private_permissions=require_private_permissions,
            ),
            header_name=header_name,
        )

    async def get_auth(self) -> MCPAuth:
        cred = await self._credentials.get_credential()
        return MCPAuth(headers={self._header_name: f"Bearer {cred.access}"})


class DatadogMCPAuth:
    """API-key auth for Datadog's hosted MCP endpoint.

    Reads credentials from environment variables by default::

        DD_API_KEY=<key> DD_APP_KEY=<key> python my_agent.py

    Or pass them explicitly::

        auth = DatadogMCPAuth(api_key="...", app_key="...")

    The ``url`` property returns the correct MCP base URL for the configured
    Datadog site (default ``datadoghq.com``; set ``site`` for EU/Gov/etc.)::

        from tools.mcp.adapter import StreamableHttpServerParams
        from tools.mcp.auth import DatadogMCPAuth

        auth = DatadogMCPAuth()
        params = StreamableHttpServerParams(url=auth.url)
        async with MCPServerConnection(params, server_name="datadog", auth=auth) as conn:
            conn.register_tools(tool_registry)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_key: str | None = None,
        site: str = "datadoghq.com",
    ) -> None:
        self._api_key = api_key or os.environ.get("DD_API_KEY", "")
        self._app_key = app_key or os.environ.get("DD_APP_KEY", "")
        self._site = site
        if not self._api_key:
            raise ValueError("api_key or DD_API_KEY environment variable required")
        if not self._app_key:
            raise ValueError("app_key or DD_APP_KEY environment variable required")

    @property
    def url(self) -> str:
        return f"https://mcp.{self._site}/"

    async def get_auth(self) -> MCPAuth:
        return MCPAuth(
            headers={
                "DD-Api-Key": self._api_key,
                "DD-Application-Key": self._app_key,
            }
        )


def merge_mcp_auth(server_params: Any, auth: MCPAuth | None) -> Any:
    """Return server params with auth applied without mutating caller input."""

    if auth is None or (not auth.headers and not auth.env):
        return server_params

    # Avoid a hard import — StreamableHttpServerParams lives in adapter.py
    # and auth.py must not depend on it (would create a circular import).
    # We duck-type on the presence of url + headers + timeout attributes.
    if (
        hasattr(server_params, "url")
        and hasattr(server_params, "headers")
        and hasattr(server_params, "timeout")
        and not isinstance(server_params, dict)
    ):
        import dataclasses as _dc

        merged_headers = {**dict(server_params.headers or {}), **dict(auth.headers)}
        return _dc.replace(server_params, headers=merged_headers)

    if isinstance(server_params, str):
        params: dict[str, Any] = {"url": server_params}
        if auth.headers:
            params["headers"] = dict(auth.headers)
        return params

    if isinstance(server_params, dict):
        merged = dict(server_params)
        if auth.headers:
            merged["headers"] = {**dict(server_params.get("headers") or {}), **dict(auth.headers)}
        if auth.env:
            merged["env"] = {**dict(server_params.get("env") or {}), **dict(auth.env)}
        return merged

    if auth.env:
        current_env = getattr(server_params, "env", None) or {}
        env = {**dict(current_env), **dict(auth.env)}
        if dataclasses.is_dataclass(server_params):
            return dataclasses.replace(server_params, env=env)
        if hasattr(server_params, "model_copy"):
            return server_params.model_copy(update={"env": env})
        try:
            return server_params.copy(update={"env": env})
        except Exception:
            pass

    if auth.headers:
        raise TypeError("MCP auth headers require a remote URL/dict transport, not stdio params")

    return server_params
