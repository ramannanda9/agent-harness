from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from tools.mcp.auth import (
    ApiKeyMCPAuth,
    BearerMCPAuth,
    BrowserOAuthMCPAuth,
    MCPAuth,
    OAuthMCPAuth,
    StaticMCPAuth,
    StreamableHttpServerParams,
    _FileTokenStorage,
    merge_mcp_auth,
)


async def test_static_mcp_auth_returns_headers_and_env():
    auth = StaticMCPAuth(
        headers={"DD_API_KEY": "api", "empty": ""},
        env={"DD_APPLICATION_KEY": "app"},
    )

    resolved = await auth.get_auth()

    assert resolved.headers == {"DD_API_KEY": "api"}
    assert resolved.env == {"DD_APPLICATION_KEY": "app"}


async def test_bearer_mcp_auth_returns_authorization_header():
    resolved = await BearerMCPAuth("token").get_auth()

    assert resolved.headers == {"Authorization": "Bearer token"}


async def test_oauth_mcp_auth_reads_auth_file(tmp_path):
    path = tmp_path / "auth.json"
    expires = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)
    path.write_text(
        json.dumps(
            {
                "datadog-mcp": {
                    "type": "oauth",
                    "access": "access-token",
                    "refresh": "refresh-token",
                    "expires": expires,
                }
            }
        )
    )
    if os.name != "nt":
        path.chmod(0o600)

    auth = OAuthMCPAuth.from_auth_file(path, provider="datadog-mcp")
    resolved = await auth.get_auth()

    assert resolved.headers == {"Authorization": "Bearer access-token"}


def test_merge_auth_into_url_string_as_headers():
    merged = merge_mcp_auth(
        "https://example.com/sse", MCPAuth(headers={"Authorization": "Bearer t"})
    )

    assert merged == {
        "url": "https://example.com/sse",
        "headers": {"Authorization": "Bearer t"},
    }


def test_merge_auth_into_dict_without_mutating_original():
    params = {
        "url": "https://example.com/sse",
        "headers": {"X-Existing": "1"},
    }

    merged = merge_mcp_auth(params, MCPAuth(headers={"Authorization": "Bearer t"}))

    assert params == {
        "url": "https://example.com/sse",
        "headers": {"X-Existing": "1"},
    }
    assert merged == {
        "url": "https://example.com/sse",
        "headers": {"X-Existing": "1", "Authorization": "Bearer t"},
    }


@dataclass(frozen=True)
class FakeStdioParams:
    command: str
    args: list[str]
    env: dict[str, str] | None = None


def test_merge_auth_env_into_dataclass_stdio_params():
    params = FakeStdioParams(command="uvx", args=["server"], env={"EXISTING": "1"})

    merged = merge_mcp_auth(params, MCPAuth(env={"TOKEN": "secret"}))

    assert params.env == {"EXISTING": "1"}
    assert merged.env == {"EXISTING": "1", "TOKEN": "secret"}


def test_headers_are_rejected_for_stdio_params():
    params = FakeStdioParams(command="uvx", args=["server"])

    with pytest.raises(TypeError, match="headers require a remote"):
        merge_mcp_auth(params, MCPAuth(headers={"Authorization": "Bearer t"}))


# ── ApiKeyMCPAuth ─────────────────────────────────────────────────────────────


async def test_api_key_auth_reads_multiple_env_vars(monkeypatch):
    monkeypatch.setenv("SVC_KEY", "key-value")
    monkeypatch.setenv("SVC_SECRET", "secret-value")

    auth = ApiKeyMCPAuth({"X-Service-Key": "SVC_KEY", "X-Service-Secret": "SVC_SECRET"})
    resolved = await auth.get_auth()

    assert resolved.headers == {"X-Service-Key": "key-value", "X-Service-Secret": "secret-value"}


async def test_api_key_auth_single_header(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")

    auth = ApiKeyMCPAuth({"Authorization": "MY_TOKEN"})
    resolved = await auth.get_auth()

    assert resolved.headers == {"Authorization": "secret"}


def test_api_key_auth_raises_on_missing_env(monkeypatch):
    monkeypatch.delenv("SVC_KEY", raising=False)
    monkeypatch.delenv("SVC_SECRET", raising=False)

    with pytest.raises(ValueError, match="SVC_KEY"):
        ApiKeyMCPAuth({"X-Service-Key": "SVC_KEY", "X-Service-Secret": "SVC_SECRET"})


def test_api_key_auth_raises_on_partial_missing_env(monkeypatch):
    monkeypatch.setenv("SVC_KEY", "present")
    monkeypatch.delenv("SVC_SECRET", raising=False)

    with pytest.raises(ValueError, match="SVC_SECRET"):
        ApiKeyMCPAuth({"X-Service-Key": "SVC_KEY", "X-Service-Secret": "SVC_SECRET"})


# ── StreamableHttpServerParams + merge_mcp_auth ──────────────────────────────


def test_merge_auth_into_streamable_http_params():
    params = StreamableHttpServerParams(url="https://mcp.datadoghq.com/")
    auth = MCPAuth(headers={"DD-Api-Key": "k1", "DD-Application-Key": "k2"})

    merged = merge_mcp_auth(params, auth)

    assert isinstance(merged, StreamableHttpServerParams)
    assert merged.url == "https://mcp.datadoghq.com/"
    assert merged.headers == {"DD-Api-Key": "k1", "DD-Application-Key": "k2"}


def test_merge_auth_into_streamable_http_params_merges_existing_headers():
    params = StreamableHttpServerParams(
        url="https://mcp.datadoghq.com/",
        headers={"X-Custom": "existing"},
    )
    auth = MCPAuth(headers={"DD-Api-Key": "k1"})

    merged = merge_mcp_auth(params, auth)

    assert merged.headers == {"X-Custom": "existing", "DD-Api-Key": "k1"}
    # original not mutated
    assert params.headers == {"X-Custom": "existing"}


def test_merge_auth_preserves_streamable_http_timeout():
    params = StreamableHttpServerParams(url="https://mcp.datadoghq.com/", timeout=60.0)
    merged = merge_mcp_auth(params, MCPAuth(headers={"DD-Api-Key": "k"}))

    assert merged.timeout == 60.0


# ── _FileTokenStorage ─────────────────────────────────────────────────────────


async def test_file_token_storage_roundtrips_tokens(tmp_path):
    from mcp.shared.auth import OAuthToken

    storage = _FileTokenStorage(tmp_path / "auth.json", "mcp:svc")
    assert await storage.get_tokens() is None

    tok = OAuthToken(access_token="A", token_type="Bearer", expires_in=3600, refresh_token="R")
    await storage.set_tokens(tok)

    loaded = await storage.get_tokens()
    assert loaded.access_token == "A"
    assert loaded.refresh_token == "R"


async def test_file_token_storage_roundtrips_client_info(tmp_path):
    from mcp.shared.auth import OAuthClientInformationFull

    storage = _FileTokenStorage(tmp_path / "auth.json", "mcp:svc")
    assert await storage.get_client_info() is None

    info = OAuthClientInformationFull(
        client_id="cid",
        client_secret="csec",
        redirect_uris=["http://127.0.0.1:8765/callback"],
    )
    await storage.set_client_info(info)

    loaded = await storage.get_client_info()
    assert loaded.client_id == "cid"
    assert loaded.client_secret == "csec"


async def test_file_token_storage_keeps_unrelated_entries(tmp_path):
    """Writing MCP entries must not clobber existing claude-code/openai-codex entries."""
    import json

    from mcp.shared.auth import OAuthToken

    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "claude-code": {"type": "oauth", "access": "X", "refresh": "Y"},
            }
        )
    )

    storage = _FileTokenStorage(path, "mcp:svc")
    await storage.set_tokens(OAuthToken(access_token="A", token_type="Bearer"))

    data = json.loads(path.read_text())
    assert data["claude-code"]["access"] == "X"  # not clobbered
    assert data["mcp:svc"]["tokens"]["access_token"] == "A"


def test_file_token_storage_writes_0600(tmp_path):
    import asyncio
    import os

    from mcp.shared.auth import OAuthToken

    if os.name == "nt":
        pytest.skip("POSIX permission semantics only")

    path = tmp_path / "auth.json"
    storage = _FileTokenStorage(path, "mcp:svc")
    asyncio.run(storage.set_tokens(OAuthToken(access_token="A", token_type="Bearer")))

    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


# ── BrowserOAuthMCPAuth ───────────────────────────────────────────────────────


async def test_browser_oauth_get_auth_returns_empty():
    """OAuth flow happens via httpx_auth; get_auth() must return an empty MCPAuth."""
    auth = BrowserOAuthMCPAuth(
        server_url="https://mcp.example.com/",
        provider_name="mcp:example",
    )
    resolved = await auth.get_auth()
    assert resolved.headers == {}
    assert resolved.env == {}


def test_browser_oauth_httpx_auth_constructs_lazily():
    """Provider should not touch the MCP SDK until httpx_auth is accessed."""
    auth = BrowserOAuthMCPAuth(
        server_url="https://mcp.example.com/",
        provider_name="mcp:example",
        scopes=["read", "write"],
    )
    assert auth._provider is None

    provider = auth.httpx_auth
    assert provider is not None
    # Second access returns the same instance — cached.
    assert auth.httpx_auth is provider


async def test_browser_oauth_static_client_id_seeds_storage(tmp_path):
    """Pre-registered client_id should be written to storage so the SDK skips
    dynamic registration (which would fail against providers that don't
    support RFC 7591)."""
    auth_file = tmp_path / "auth.json"
    auth = BrowserOAuthMCPAuth(
        server_url="https://mcp.example.com/",
        provider_name="mcp:example",
        client_id="my-client-id",
        client_secret="my-secret",
        auth_file=auth_file,
    )
    # Trigger lazy construction.
    _ = auth.httpx_auth

    storage = _FileTokenStorage(auth_file, "mcp:example")
    info = await storage.get_client_info()
    assert info is not None
    assert info.client_id == "my-client-id"
    assert info.client_secret == "my-secret"


def test_browser_oauth_does_not_reseed_matching_client_id(tmp_path):
    """When storage already has client_info matching the supplied client_id,
    leave it alone (avoids touching the file on every construct)."""
    import json

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "mcp:example": {
                    "type": "mcp-oauth",
                    "client_info": {
                        "client_id": "static-client-id",
                        "redirect_uris": ["http://127.0.0.1:8765/callback"],
                    },
                    "tokens": {"access_token": "keep-me", "token_type": "Bearer"},
                }
            }
        )
    )

    auth = BrowserOAuthMCPAuth(
        server_url="https://mcp.example.com/",
        provider_name="mcp:example",
        client_id="static-client-id",
        auth_file=auth_file,
    )
    _ = auth.httpx_auth

    data = json.loads(auth_file.read_text())
    assert data["mcp:example"]["tokens"]["access_token"] == "keep-me"


def test_browser_oauth_reseeds_stale_entry_missing_client_info(tmp_path):
    """A previous run without client_id can leave a stale entry. We must
    write client_info into it so the SDK skips dynamic registration."""
    import json

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({"mcp:example": {"type": "mcp-oauth"}})  # no client_info
    )

    auth = BrowserOAuthMCPAuth(
        server_url="https://mcp.example.com/",
        provider_name="mcp:example",
        client_id="static-client-id",
        auth_file=auth_file,
    )
    _ = auth.httpx_auth

    data = json.loads(auth_file.read_text())
    assert data["mcp:example"]["client_info"]["client_id"] == "static-client-id"
