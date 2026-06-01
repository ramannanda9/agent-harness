"""Authentication helpers and connection parameter types for MCP servers."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness.llm.auth import AuthFileOAuthProvider


@dataclass
class StreamableHttpServerParams:
    """Connection parameters for an MCP server using the streamable-HTTP transport.

    Headers supplied here are merged with those from the MCPAuthProvider before
    the connection is opened.

    Example::

        # API-key auth
        auth = ApiKeyMCPAuth({"X-Api-Key": "MY_SERVICE_KEY"})
        # or OAuth from auth file
        # auth = OAuthMCPAuth.from_auth_file("~/.agent-harness/auth/auth.json", provider="svc")
        params = StreamableHttpServerParams(url="https://mcp.example.com/")
        async with MCPServerConnection(params, auth=auth) as conn:
            ...
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    sse_read_timeout: float = 300.0


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


# ── Browser-based OAuth ──────────────────────────────────────────────────────


class BrowserOAuthMCPAuth:
    """OAuth 2.0 + PKCE for MCP servers via a localhost browser flow.

    Delegates the OAuth dance to ``mcp.client.auth.OAuthClientProvider`` — it
    handles PKCE, dynamic client registration (RFC 7591), token refresh, and
    server metadata discovery (RFC 8414). This class wires it up with:

      - a localhost callback server (see :mod:`harness.oauth_browser`),
      - automatic browser launch with print-URL fallback,
      - token persistence to the same ``auth.json`` file used by the
        Claude Code / OpenAI Codex login flows (so MCP and LLM tokens
        live in one place).

    First connect attempt opens the browser; subsequent runs read the
    cached tokens and refresh them transparently when expired.

    Most commercial OAuth providers require a pre-registered app and reject
    requests without a ``client_id``. Register your app at the provider's
    developer console (callback URL: ``http://127.0.0.1:8765/callback`` by
    default) and pass the resulting ``client_id`` (and ``client_secret`` if
    the provider issues one):

        auth = BrowserOAuthMCPAuth(
            server_url="https://mcp.example.com/",
            provider_name="mcp:example",
            client_id="abc123",                  # from provider's dev console
            client_secret="shh",                 # optional; PKCE-only flows omit
            scopes=["read", "write"],
        )
        params = StreamableHttpServerParams(url="https://mcp.example.com/")
        async with MCPServerConnection(params, auth=auth) as conn:
            conn.register_tools(tool_registry)

    Servers that support RFC 7591 dynamic client registration can be used
    without supplying ``client_id`` — the SDK registers a fresh client on
    first connect and persists it to ``auth_file``.
    """

    def __init__(
        self,
        *,
        server_url: str,
        provider_name: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        client_name: str = "agent-harness",
        scopes: list[str] | None = None,
        auth_file: str | os.PathLike[str] | None = None,
        callback_port: int = 8765,
        callback_path: str = "/callback",
        callback_timeout: float = 300.0,
        open_browser: bool = True,
    ) -> None:
        from pathlib import Path

        self._server_url = server_url
        self._client_name = client_name
        self._client_id = client_id
        self._client_secret = client_secret
        self._scopes = list(scopes or [])
        self._auth_file = (
            Path(auth_file).expanduser()
            if auth_file
            else Path("~/.agent-harness/auth/auth.json").expanduser()
        )
        self._provider_name = provider_name
        self._callback_port = callback_port
        self._callback_path = callback_path
        self._callback_timeout = callback_timeout
        self._open_browser = open_browser
        self._provider: Any = None  # lazily constructed OAuthClientProvider

    @property
    def httpx_auth(self) -> Any:
        """The ``httpx.Auth`` instance for streamable-HTTP / SSE transports.

        ``MCPServerConnection`` detects this attribute and passes the
        provider straight to the transport client's ``auth=`` parameter,
        bypassing the static-header path.
        """
        if self._provider is None:
            self._provider = self._build_provider()
        return self._provider

    async def get_auth(self) -> MCPAuth:
        """Satisfy the ``MCPAuthProvider`` protocol; OAuth uses ``httpx_auth``."""
        return MCPAuth()

    def _build_provider(self) -> Any:
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata

        storage = _FileTokenStorage(self._auth_file, self._provider_name)
        port = self._callback_port
        redirect_uri = f"http://127.0.0.1:{port}{self._callback_path}"
        client_metadata = OAuthClientMetadata(
            client_name=self._client_name,
            redirect_uris=[redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=" ".join(self._scopes) if self._scopes else None,
        )

        # Most commercial OAuth providers don't support RFC 7591 dynamic
        # client registration; you register an app manually in their UI and
        # get a static client_id. Seed storage so the SDK skips registration
        # and goes straight to the authorisation URL.
        #
        # We seed when client_info is missing (not just when the whole entry
        # is missing) so that a previous failed attempt without client_id
        # — which leaves a stale {"type": "mcp-oauth", "tokens": {...}}
        # entry — gets the static client_info written into it on next run.
        if self._client_id:
            entry = storage._read_entry() or {"type": "mcp-oauth"}
            existing_info = entry.get("client_info") or {}
            if existing_info.get("client_id") != self._client_id:
                entry["client_info"] = OAuthClientInformationFull(
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                    redirect_uris=[redirect_uri],
                ).model_dump(mode="json", exclude_none=True)
                storage._write_entry(entry)

        async def redirect_handler(url: str) -> None:
            from harness.oauth_browser import open_or_print_url

            prefix = (
                "Authorize this client in your browser:"
                if self._open_browser
                else "Open this URL in your browser:"
            )
            if self._open_browser:
                open_or_print_url(url, prefix=prefix)
            else:
                print(f"{prefix} {url}")

        async def callback_handler() -> tuple[str, str | None]:
            from harness.oauth_browser import wait_for_oauth_callback

            return await wait_for_oauth_callback(
                port=port,
                path=self._callback_path,
                timeout=self._callback_timeout,
            )

        return OAuthClientProvider(
            server_url=self._server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=self._callback_timeout,
        )


class _FileTokenStorage:
    """MCP SDK ``TokenStorage`` backed by the agent-harness auth.json file.

    Stores both the ``OAuthToken`` and ``OAuthClientInformationFull`` under a
    single entry keyed by ``provider_name``. Layout::

        {
          "mcp:example": {
            "type": "mcp-oauth",
            "tokens": {...},        # pydantic OAuthToken dump
            "client_info": {...},   # pydantic OAuthClientInformationFull dump
          }
        }

    The ``type`` discriminator keeps MCP entries separate from the legacy
    LLM-provider ``type="oauth"`` entries so the two storage layers don't
    collide on shared keys.
    """

    def __init__(self, path: Any, provider_name: str) -> None:
        from pathlib import Path

        self._path = Path(path).expanduser()
        self._provider = provider_name

    async def get_tokens(self) -> Any:
        from mcp.shared.auth import OAuthToken

        entry = self._read_entry()
        tokens = entry.get("tokens") if entry else None
        if not tokens:
            return None
        return OAuthToken.model_validate(tokens)

    async def set_tokens(self, tokens: Any) -> None:
        entry = self._read_entry() or {"type": "mcp-oauth"}
        entry["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._write_entry(entry)

    async def get_client_info(self) -> Any:
        from mcp.shared.auth import OAuthClientInformationFull

        entry = self._read_entry()
        info = entry.get("client_info") if entry else None
        if not info:
            return None
        return OAuthClientInformationFull.model_validate(info)

    async def set_client_info(self, client_info: Any) -> None:
        entry = self._read_entry() or {"type": "mcp-oauth"}
        entry["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write_entry(entry)

    # ── persistence helpers ───────────────────────────────────────────────

    def _read_entry(self) -> dict[str, Any] | None:
        import json

        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text() or "{}")
        except (OSError, ValueError):
            return None
        entry = data.get(self._provider)
        return entry if isinstance(entry, dict) else None

    def _write_entry(self, entry: dict[str, Any]) -> None:
        import json

        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text() or "{}")
            except (OSError, ValueError):
                data = {}
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[self._provider] = entry
        self._path.write_text(json.dumps(data, indent=2))
        try:
            self._path.chmod(0o600)
        except OSError:
            pass


def merge_mcp_auth(server_params: Any, auth: MCPAuth | None) -> Any:
    """Return server params with auth applied without mutating caller input."""

    if auth is None or (not auth.headers and not auth.env):
        return server_params

    if isinstance(server_params, StreamableHttpServerParams):
        merged_headers = {**dict(server_params.headers or {}), **dict(auth.headers)}
        return dataclasses.replace(server_params, headers=merged_headers)

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
