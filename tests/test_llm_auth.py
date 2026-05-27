from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from harness.llm.auth import (
    AnthropicClaudeCodeOAuthClient,
    AuthFileOAuthProvider,
    AuthorizationCodeLogin,
    CommandTokenProvider,
    DeviceCode,
    FileTokenProvider,
    OAuthCredential,
    OAuthPending,
    OpenAICodexOAuthClient,
    StaticTokenProvider,
    decode_jwt_payload,
    extract_openai_account_id,
    parse_authorization_callback,
    parse_token_output,
)


def test_parse_token_output_accepts_plain_token():
    token = parse_token_output("abc123")
    assert token.value == "abc123"
    assert token.token_type == "Bearer"
    assert token.expires_at is None


def test_parse_token_output_accepts_json_with_expires_in():
    token = parse_token_output('{"access_token":"abc","expires_in":120,"token_type":"Bearer"}')
    assert token.value == "abc"
    assert token.expires_at is not None
    assert token.expires_at > datetime.now(timezone.utc)


def test_parse_authorization_callback_accepts_url_and_code_state():
    assert parse_authorization_callback("https://x/callback?code=abc&state=def") == ("abc", "def")
    assert parse_authorization_callback("abc#def") == ("abc", "def")
    assert parse_authorization_callback("code=abc&state=def") == ("abc", "def")
    assert parse_authorization_callback("nope") is None


async def test_static_token_provider_returns_fixed_token():
    provider = StaticTokenProvider("fixed")
    assert (await provider.get_token()).value == "fixed"
    assert (await provider.get_token(force_refresh=True)).value == "fixed"


async def test_file_token_provider_requires_private_permissions(tmp_path):
    path = tmp_path / "token.json"
    path.write_text(json.dumps({"access_token": "secret"}))
    if os.name != "nt":
        path.chmod(0o644)
        with pytest.raises(PermissionError):
            await FileTokenProvider(path).get_token()

        path.chmod(0o600)

    token = await FileTokenProvider(path).get_token()
    assert token.value == "secret"


async def test_command_token_provider_caches_until_expired(monkeypatch):
    calls = {"n": 0}

    class _Proc:
        returncode = 0

        async def communicate(self):
            calls["n"] += 1
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            return json.dumps(
                {"access_token": f"tok-{calls['n']}", "expires_at": expires_at}
            ).encode(), b""

    async def fake_exec(*_args, **_kwargs):
        return _Proc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    provider = CommandTokenProvider(["helper", "token"])

    assert (await provider.get_token()).value == "tok-1"
    assert (await provider.get_token()).value == "tok-1"
    assert (await provider.get_token(force_refresh=True)).value == "tok-2"


def test_decode_jwt_payload_and_extract_account_id():
    import base64

    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acct_123",
            "chatgpt_plan_type": "plus",
        }
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    token = f"header.{encoded}.sig"

    assert decode_jwt_payload(token) == payload
    assert extract_openai_account_id(token) == "acct_123"


async def test_auth_file_oauth_provider_reads_pi_shaped_entry(tmp_path):
    path = tmp_path / "auth.json"
    expires = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)
    path.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": "access-token",
                    "refresh": "refresh-token",
                    "expires": expires,
                    "accountId": "acct_123",
                }
            }
        )
    )
    if os.name != "nt":
        path.chmod(0o600)

    provider = AuthFileOAuthProvider(path, provider="openai-codex")
    cred = await provider.get_credential()

    assert cred.access == "access-token"
    assert cred.refresh == "refresh-token"
    assert cred.account_id == "acct_123"
    assert (await provider.get_token()).value == "access-token"


async def test_auth_file_oauth_provider_refreshes_and_persists(tmp_path):
    path = tmp_path / "auth.json"
    expired = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000)
    path.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": "old-access",
                    "refresh": "old-refresh",
                    "expires": expired,
                    "accountId": "acct_old",
                }
            }
        )
    )
    if os.name != "nt":
        path.chmod(0o600)

    async def refresher(old: OAuthCredential) -> OAuthCredential:
        assert old.access == "old-access"
        return OAuthCredential(
            provider="openai-codex",
            access="new-access",
            refresh="new-refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
            account_id="acct_new",
        )

    provider = AuthFileOAuthProvider(path, provider="openai-codex", refresher=refresher)
    cred = await provider.get_credential()

    assert cred.access == "new-access"
    data = json.loads(path.read_text())
    assert data["openai-codex"]["access"] == "new-access"
    assert data["openai-codex"]["refresh"] == "new-refresh"
    assert data["openai-codex"]["accountId"] == "acct_new"


class _OAuthResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _OAuthHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


async def test_openai_codex_oauth_client_requests_device_code():
    http = _OAuthHttpClient(
        [
            _OAuthResponse(
                200,
                {
                    "device_auth_id": "dev",
                    "user_code": "ABCD-1234",
                    "expires_in": 900,
                    "interval": 2,
                },
            )
        ]
    )
    client = OpenAICodexOAuthClient(http_client=http)

    device = await client.request_device_code()

    assert device == DeviceCode(
        device_auth_id="dev",
        user_code="ABCD-1234",
        verification_uri="https://auth.openai.com/codex/device",
        expires_in=900,
        interval=2,
    )
    assert http.calls[0]["url"].endswith("/api/accounts/deviceauth/usercode")


async def test_openai_codex_oauth_client_refreshes_token():
    http = _OAuthHttpClient(
        [
            _OAuthResponse(
                200,
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                },
            )
        ]
    )
    client = OpenAICodexOAuthClient(http_client=http)
    old = OAuthCredential(provider="openai-codex", access="old", refresh="old-refresh")

    refreshed = await client.refresh(old)

    assert refreshed.access == "new-access"
    assert refreshed.refresh == "new-refresh"
    assert http.calls[0]["url"].endswith("/oauth/token")
    assert http.calls[0]["data"]["grant_type"] == "refresh_token"


async def test_openai_codex_oauth_client_device_poll_pending_then_success(monkeypatch):
    calls = {"sleep": 0}

    async def fake_sleep(_seconds):
        calls["sleep"] += 1

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    http = _OAuthHttpClient(
        [
            _OAuthResponse(403, {}),
            _OAuthResponse(
                200,
                {
                    "authorization_code": "auth-code",
                    "code_challenge": "challenge",
                    "code_verifier": "verifier",
                },
            ),
            _OAuthResponse(
                200,
                {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "expires_in": 3600,
                },
            ),
        ]
    )
    client = OpenAICodexOAuthClient(http_client=http)
    device = DeviceCode("dev", "ABCD", "https://example", 900, interval=1)

    cred = await client.poll_device_code(device)

    assert cred.access == "access"
    assert calls["sleep"] == 1
    assert http.calls[1]["json"] == {"device_auth_id": "dev", "user_code": "ABCD"}
    assert http.calls[2]["url"].endswith("/oauth/token")
    assert http.calls[2]["data"]["grant_type"] == "authorization_code"
    assert http.calls[2]["data"]["code_verifier"] == "verifier"


async def test_openai_codex_oauth_pending_exception():
    http = _OAuthHttpClient([_OAuthResponse(400, {"error": "authorization_pending"})])
    client = OpenAICodexOAuthClient(http_client=http)
    with pytest.raises(OAuthPending):
        await client._exchange_device_code("dev")


def test_anthropic_claude_code_oauth_builds_authorize_url():
    client = AnthropicClaudeCodeOAuthClient()
    login = client.begin_login()

    assert "https://claude.ai/oauth/authorize?" in login.url
    assert "client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e" in login.url
    assert "code_challenge=" in login.url
    assert login.verifier
    assert login.state


async def test_anthropic_claude_code_oauth_exchanges_authorization_code():
    http = _OAuthHttpClient(
        [
            _OAuthResponse(
                200,
                {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "expires_in": 3600,
                },
            )
        ]
    )
    client = AnthropicClaudeCodeOAuthClient(http_client=http)
    login = AuthorizationCodeLogin(
        url="https://example",
        state="state",
        verifier="verifier",
        redirect_uri="https://platform.claude.com/oauth/code/callback",
    )

    cred = await client.finish_login(login, "https://local/callback?code=code&state=state")

    assert cred.provider == "claude-code"
    assert cred.access == "access"
    assert cred.refresh == "refresh"
    assert http.calls[0]["url"] == "https://platform.claude.com/v1/oauth/token"
    assert http.calls[0]["json"]["grant_type"] == "authorization_code"
    assert http.calls[0]["json"]["code_verifier"] == "verifier"


async def test_anthropic_claude_code_oauth_refreshes_token():
    http = _OAuthHttpClient(
        [
            _OAuthResponse(
                200,
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                },
            )
        ]
    )
    client = AnthropicClaudeCodeOAuthClient(http_client=http)
    old = OAuthCredential(provider="claude-code", access="old", refresh="old-refresh")

    refreshed = await client.refresh(old)

    assert refreshed.access == "new-access"
    assert refreshed.refresh == "new-refresh"
    assert http.calls[0]["json"]["grant_type"] == "refresh_token"
