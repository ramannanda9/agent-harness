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


class ApiKeyMCPAuth:
    """Header-based auth where values are read from environment variables.

    Maps HTTP header names to environment variable names. Suitable for any
    remote MCP server that authenticates via API keys in request headers.

    Example — Datadog::

        auth = ApiKeyMCPAuth({
            "DD-Api-Key": "DD_API_KEY",
            "DD-Application-Key": "DD_APP_KEY",
        })
        params = StreamableHttpServerParams(url="https://mcp.datadoghq.com/")
        async with MCPServerConnection(params, auth=auth) as conn:
            ...

    Example — single-key service::

        auth = ApiKeyMCPAuth({"Authorization": "MY_SERVICE_TOKEN"})

    Raises ``ValueError`` at construction time if any mapped environment
    variable is absent or empty, so misconfiguration fails fast.
    """

    def __init__(self, header_env_map: dict[str, str]) -> None:
        missing = [env for env in header_env_map.values() if not os.environ.get(env)]
        if missing:
            raise ValueError(f"Missing or empty environment variable(s): {', '.join(missing)}")
        self._map = dict(header_env_map)

    async def get_auth(self) -> MCPAuth:
        return MCPAuth(headers={header: os.environ[env] for header, env in self._map.items()})


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
