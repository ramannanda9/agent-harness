from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from tools.mcp.auth import BearerMCPAuth, MCPAuth, OAuthMCPAuth, StaticMCPAuth, merge_mcp_auth


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
