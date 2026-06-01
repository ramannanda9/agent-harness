from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from tools.mcp.auth import (
    ApiKeyMCPAuth,
    BearerMCPAuth,
    MCPAuth,
    OAuthMCPAuth,
    StaticMCPAuth,
    StreamableHttpServerParams,
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
