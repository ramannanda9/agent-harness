from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from tools.mcp.adapter import StreamableHttpServerParams
from tools.mcp.auth import (
    BearerMCPAuth,
    DatadogMCPAuth,
    MCPAuth,
    OAuthMCPAuth,
    StaticMCPAuth,
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


# ── DatadogMCPAuth ────────────────────────────────────────────────────────────


async def test_datadog_auth_reads_env(monkeypatch):
    monkeypatch.setenv("DD_API_KEY", "test-api-key")
    monkeypatch.setenv("DD_APP_KEY", "test-app-key")

    auth = DatadogMCPAuth()
    resolved = await auth.get_auth()

    assert resolved.headers == {
        "DD-Api-Key": "test-api-key",
        "DD-Application-Key": "test-app-key",
    }


async def test_datadog_auth_explicit_keys():
    auth = DatadogMCPAuth(api_key="ak", app_key="appk")
    resolved = await auth.get_auth()

    assert resolved.headers["DD-Api-Key"] == "ak"
    assert resolved.headers["DD-Application-Key"] == "appk"


def test_datadog_auth_url_default_site():
    auth = DatadogMCPAuth(api_key="k", app_key="k")
    assert auth.url == "https://mcp.datadoghq.com/"


def test_datadog_auth_url_custom_site():
    auth = DatadogMCPAuth(api_key="k", app_key="k", site="us5.datadoghq.com")
    assert auth.url == "https://mcp.us5.datadoghq.com/"


def test_datadog_auth_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("DD_API_KEY", raising=False)
    monkeypatch.delenv("DD_APP_KEY", raising=False)

    with pytest.raises(ValueError, match="DD_API_KEY"):
        DatadogMCPAuth()


def test_datadog_auth_raises_without_app_key(monkeypatch):
    monkeypatch.delenv("DD_APP_KEY", raising=False)

    with pytest.raises(ValueError, match="DD_APP_KEY"):
        DatadogMCPAuth(api_key="k")


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
