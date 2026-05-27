"""Credential helpers for LLM adapters.

The harness deliberately keeps provider auth out of agents. Adapters ask a
CredentialProvider for a valid token before each request; providers may use
static API keys, OS/keyring-backed helpers, command helpers, or official CLI
sessions. This module avoids browser-token scraping and stores no secrets by
default.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlencode, urlparse

# Public Codex OAuth client id observed in Codex-compatible tooling. This is
# not documented as a stable OpenAI Platform API contract, so callers can
# override it with AGENT_HARNESS_OPENAI_CODEX_CLIENT_ID.
OPENAI_CODEX_CLIENT_ID = os.environ.get(
    "AGENT_HARNESS_OPENAI_CODEX_CLIENT_ID",
    "app_EMoamEEZ73f0CkXaXp7hrann",
)
ANTHROPIC_CLAUDE_CODE_CLIENT_ID = os.environ.get(
    "AGENT_HARNESS_ANTHROPIC_CLAUDE_CODE_CLIENT_ID",
    "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
)
ANTHROPIC_CLAUDE_CODE_SCOPES = " ".join(
    [
        "org:create_api_key",
        "user:profile",
        "user:inference",
        "user:sessions:claude_code",
        "user:mcp_servers",
        "user:file_upload",
    ]
)


@dataclass(frozen=True)
class AccessToken:
    value: str
    expires_at: datetime | None = None
    token_type: str = "Bearer"

    def is_expired(self, *, skew_seconds: int = 60) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) + timedelta(seconds=skew_seconds) >= self.expires_at


class CredentialProvider(Protocol):
    async def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        """Return a usable access token, refreshing if needed."""


class StaticTokenProvider:
    """Credential provider for fixed API keys or bearer tokens."""

    def __init__(
        self,
        token: str,
        *,
        token_type: str = "Bearer",
        expires_at: datetime | None = None,
    ) -> None:
        if not token or not token.strip():
            raise ValueError("token must be non-empty")
        self._token = AccessToken(token.strip(), expires_at=expires_at, token_type=token_type)

    async def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        return self._token


class FileTokenProvider:
    """Load a token from a JSON file.

    Expected JSON keys:
      - access_token, token, or api_key
      - optional token_type
      - optional expires_at (ISO-8601 or epoch seconds)

    The file is expected to be user-private. On POSIX, group/world-readable
    files are rejected by default.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        require_private_permissions: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        self.require_private_permissions = require_private_permissions
        self._cached: AccessToken | None = None

    async def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        if self._cached is not None and not force_refresh and not self._cached.is_expired():
            return self._cached
        self._validate_permissions()
        data = json.loads(self.path.read_text())
        self._cached = _token_from_payload(data)
        return self._cached

    def _validate_permissions(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if os.name == "nt" or not self.require_private_permissions:
            return
        mode = stat.S_IMODE(self.path.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(f"{self.path} must not be readable or writable by group/other")


class CommandTokenProvider:
    """Run a local command that prints a token payload.

    This is the safest extension point for provider-specific OAuth helpers:
    the helper owns login, refresh, and token storage; agent-harness only sees
    a short-lived token. Output may be JSON or a plain token string.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 15.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.env = dict(env or {})
        self._cached: AccessToken | None = None

    async def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        if self._cached is not None and not force_refresh and not self._cached.is_expired():
            return self._cached

        proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"token command timed out: {self.command[0]}") from None

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"token command failed ({proc.returncode}): {err}")

        text = stdout.decode(errors="replace").strip()
        if not text:
            raise RuntimeError("token command produced no output")
        self._cached = parse_token_output(text)
        return self._cached


@dataclass(frozen=True)
class OAuthCredential:
    provider: str
    access: str
    refresh: str | None = None
    expires_at: datetime | None = None
    account_id: str | None = None

    def is_expired(self, *, skew_seconds: int = 60) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) + timedelta(seconds=skew_seconds) >= self.expires_at


class AuthFileOAuthProvider:
    """Read Pi-style OAuth credentials from an auth.json file.

    Example shape:
        {
          "openai-codex": {
            "type": "oauth",
            "access": "...",
            "refresh": "...",
            "expires": 1762857415123,
            "accountId": "..."
          }
        }

    If a credential is expired and a refresher is supplied, the refresher is
    called and the updated credential is persisted with 0600 permissions.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        provider: str,
        refresher: Callable[[OAuthCredential], Awaitable[OAuthCredential]] | None = None,
        require_private_permissions: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        self.provider = provider
        self.refresher = refresher
        self.require_private_permissions = require_private_permissions
        self._cached: OAuthCredential | None = None

    async def get_credential(self, *, force_refresh: bool = False) -> OAuthCredential:
        cred = None if force_refresh else self._cached
        if cred is None:
            cred = self._read_credential()
        if (force_refresh or cred.is_expired()) and self.refresher is not None:
            cred = await self.refresher(cred)
            self._write_credential(cred)
        self._cached = cred
        return cred

    async def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        cred = await self.get_credential(force_refresh=force_refresh)
        return AccessToken(cred.access, expires_at=cred.expires_at)

    def _read_credential(self) -> OAuthCredential:
        _validate_private_file(
            self.path, require_private_permissions=self.require_private_permissions
        )
        data = json.loads(self.path.read_text())
        entry = data.get(self.provider)
        if not isinstance(entry, dict):
            raise KeyError(f"{self.path} has no {self.provider!r} OAuth entry")
        if entry.get("type") != "oauth":
            raise ValueError(f"{self.provider!r} auth entry is not type='oauth'")
        access = entry.get("access") or entry.get("access_token")
        if not isinstance(access, str) or not access.strip():
            raise ValueError(f"{self.provider!r} OAuth entry has no access token")
        refresh = entry.get("refresh") or entry.get("refresh_token")
        expires_at = _parse_oauth_expires(entry)
        account_id = (
            entry.get("accountId") or entry.get("account_id") or extract_openai_account_id(access)
        )
        return OAuthCredential(
            provider=self.provider,
            access=access.strip(),
            refresh=refresh.strip() if isinstance(refresh, str) else None,
            expires_at=expires_at,
            account_id=account_id if isinstance(account_id, str) else None,
        )

    def _write_credential(self, cred: OAuthCredential) -> None:
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        data[self.provider] = {
            "type": "oauth",
            "access": cred.access,
            "refresh": cred.refresh,
            "expires": _expires_to_millis(cred.expires_at),
            "accountId": cred.account_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        if os.name != "nt":
            self.path.chmod(0o600)

    def clear(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        data.pop(self.provider, None)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        if os.name != "nt":
            self.path.chmod(0o600)


@dataclass(frozen=True)
class DeviceCode:
    device_auth_id: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int = 5


class OpenAICodexOAuthClient:
    """OAuth helper for OpenAI Codex subscription auth.

    The endpoints mirror the Codex/ChatGPT device flow used by Codex-family
    tools. They are intentionally isolated here so the rest of the harness only
    depends on `OAuthCredential`.
    """

    def __init__(
        self,
        *,
        client_id: str = OPENAI_CODEX_CLIENT_ID,
        auth_base_url: str = "https://auth.openai.com",
        http_client: Any | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.client_id = client_id
        self.auth_base_url = auth_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        self.timeout_seconds = timeout_seconds

    async def request_device_code(self) -> DeviceCode:
        client = await self._get_client()
        response = await client.post(
            f"{self.auth_base_url}/api/accounts/deviceauth/usercode",
            json={
                "client_id": self.client_id,
                "scope": "openid profile email offline_access",
            },
        )
        response.raise_for_status()
        data = response.json()
        device_auth_id = data.get("device_auth_id") or data.get("device_code")
        user_code = data.get("user_code") or data.get("usercode")
        if not isinstance(device_auth_id, str) or not device_auth_id:
            raise ValueError("OpenAI Codex device auth response has no device_auth_id")
        if not isinstance(user_code, str) or not user_code:
            raise ValueError("OpenAI Codex device auth response has no user_code")
        verification_uri = (
            data.get("verification_uri")
            or data.get("verification_url")
            or f"{self.auth_base_url}/codex/device"
        )
        return DeviceCode(
            device_auth_id=device_auth_id,
            user_code=user_code,
            verification_uri=str(verification_uri),
            expires_in=int(data.get("expires_in") or 900),
            interval=int(data.get("interval") or 5),
        )

    async def poll_device_code(self, device: DeviceCode) -> OAuthCredential:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=device.expires_in)
        interval = device.interval
        while datetime.now(timezone.utc) < deadline:
            try:
                return await self._exchange_device_code(device)
            except OAuthPending:
                await asyncio.sleep(interval)
            except OAuthSlowDown:
                interval += 5
                await asyncio.sleep(interval)
        raise TimeoutError("OpenAI Codex device login expired")

    async def refresh(self, cred: OAuthCredential) -> OAuthCredential:
        if not cred.refresh:
            raise ValueError("OAuth credential has no refresh token")
        client = await self._get_client()
        response = await client.post(
            f"{self.auth_base_url}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": cred.refresh,
                "client_id": self.client_id,
            },
        )
        response.raise_for_status()
        return _oauth_credential_from_token_response(
            response.json(),
            provider=cred.provider,
            fallback_refresh=cred.refresh,
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _exchange_device_code(self, device: DeviceCode | str) -> OAuthCredential:
        if isinstance(device, DeviceCode):
            device_auth_id = device.device_auth_id
            user_code = device.user_code
        else:
            device_auth_id = device
            user_code = ""
        client = await self._get_client()
        response = await client.post(
            f"{self.auth_base_url}/api/accounts/deviceauth/token",
            json={
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            },
        )
        if response.status_code in (403, 404):
            raise OAuthPending()
        if response.status_code == 400:
            data = _response_json(response)
            err = data.get("error")
            if err in {"authorization_pending", "pending"}:
                raise OAuthPending()
            if err == "slow_down":
                raise OAuthSlowDown()
        response.raise_for_status()
        data = response.json()
        if "access_token" in data or "access" in data:
            return _oauth_credential_from_token_response(data, provider="openai-codex")
        code = data.get("authorization_code") or data.get("code")
        code_verifier = data.get("code_verifier")
        if not isinstance(code, str) or not isinstance(code_verifier, str):
            raise ValueError("OpenAI Codex device token response has no authorization code")
        return await self._exchange_authorization_code(code, code_verifier)

    async def _exchange_authorization_code(
        self,
        code: str,
        code_verifier: str,
    ) -> OAuthCredential:
        client = await self._get_client()
        redirect_uri = f"{self.auth_base_url}/deviceauth/callback"
        response = await client.post(
            f"{self.auth_base_url}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code >= 400:
            detail = _response_error_detail(response)
            raise RuntimeError(f"OpenAI Codex OAuth token exchange failed: {detail}")
        return _oauth_credential_from_token_response(response.json(), provider="openai-codex")

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    'httpx package not installed. Run: pip install -e ".[http]"'
                ) from e
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client


class OAuthPending(Exception):
    pass


class OAuthSlowDown(Exception):
    pass


@dataclass(frozen=True)
class AuthorizationCodeLogin:
    url: str
    state: str
    verifier: str
    redirect_uri: str


class AnthropicClaudeCodeOAuthClient:
    """OAuth helper for Claude Pro/Max subscription auth."""

    def __init__(
        self,
        *,
        client_id: str = ANTHROPIC_CLAUDE_CODE_CLIENT_ID,
        authorize_url: str = "https://claude.ai/oauth/authorize",
        token_url: str = "https://platform.claude.com/v1/oauth/token",
        redirect_uri: str = "https://platform.claude.com/oauth/code/callback",
        http_client: Any | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.client_id = client_id
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.redirect_uri = redirect_uri
        self._client = http_client
        self._owns_client = http_client is None
        self.timeout_seconds = timeout_seconds

    def begin_login(self) -> AuthorizationCodeLogin:
        verifier = _pkce_verifier()
        state = secrets.token_hex(16)
        params = {
            "code": "true",
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": ANTHROPIC_CLAUDE_CODE_SCOPES,
            "code_challenge": _pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "state": state,
        }
        return AuthorizationCodeLogin(
            url=f"{self.authorize_url}?{urlencode(params)}",
            state=state,
            verifier=verifier,
            redirect_uri=self.redirect_uri,
        )

    async def finish_login(
        self, login: AuthorizationCodeLogin, callback_input: str
    ) -> OAuthCredential:
        parsed = parse_authorization_callback(callback_input)
        if parsed is None:
            raise ValueError("Could not parse authorization callback input")
        code, state = parsed
        if state != login.state:
            raise ValueError("OAuth state mismatch")
        client = await self._get_client()
        response = await client.post(
            self.token_url,
            headers={"Content-Type": "application/json"},
            json={
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "code": code,
                "state": state,
                "redirect_uri": login.redirect_uri,
                "code_verifier": login.verifier,
            },
        )
        if response.status_code >= 400:
            detail = _response_error_detail(response)
            raise RuntimeError(f"Claude Code OAuth token exchange failed: {detail}")
        return _oauth_credential_from_token_response(response.json(), provider="claude-code")

    async def refresh(self, cred: OAuthCredential) -> OAuthCredential:
        if not cred.refresh:
            raise ValueError("OAuth credential has no refresh token")
        client = await self._get_client()
        response = await client.post(
            self.token_url,
            headers={"Content-Type": "application/json"},
            json={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": cred.refresh,
            },
        )
        if response.status_code >= 400:
            detail = _response_error_detail(response)
            raise RuntimeError(f"Claude Code OAuth refresh failed: {detail}")
        return _oauth_credential_from_token_response(
            response.json(),
            provider=cred.provider,
            fallback_refresh=cred.refresh,
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    'httpx package not installed. Run: pip install -e ".[http]"'
                ) from e
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client


def parse_token_output(text: str) -> AccessToken:
    """Parse JSON token helper output, falling back to a plain token string."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return AccessToken(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("token command JSON output must be an object")
    return _token_from_payload(payload)


def parse_authorization_callback(text: str) -> tuple[str, str] | None:
    value = text.strip()
    if not value:
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.query:
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        if code and state:
            return code, state
    if "#" in value:
        code, state = value.split("#", 1)
        if code and state:
            return code, state
    params = parse_qs(value)
    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]
    if code and state:
        return code, state
    return None


def default_auth_dir() -> Path:
    return Path(os.environ.get("AGENT_HARNESS_AUTH_DIR", "~/.agent-harness/auth")).expanduser()


def default_auth_file() -> Path:
    return default_auth_dir() / "auth.json"


def extract_openai_account_id(access_token: str) -> str | None:
    payload = decode_jwt_payload(access_token)
    auth_claim = payload.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        for key in ("chatgpt_account_id", "account_id", "user_id", "chatgpt_user_id"):
            value = auth_claim.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("chatgpt_account_id", "account_id", "sub"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(f"{payload}{padding}")
        decoded = json.loads(raw)
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _token_from_payload(payload: dict[str, Any]) -> AccessToken:
    value = payload.get("access_token") or payload.get("token") or payload.get("api_key")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("token payload must include access_token, token, or api_key")
    token_type = str(payload.get("token_type") or "Bearer")
    expires_at = _parse_expires_at(payload)
    return AccessToken(value.strip(), expires_at=expires_at, token_type=token_type)


def _validate_private_file(
    path: Path,
    *,
    require_private_permissions: bool,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if os.name == "nt" or not require_private_permissions:
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"{path} must not be readable or writable by group/other")


def _parse_oauth_expires(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("expires_at")
    if raw is None:
        raw = payload.get("expires")
    if raw is None:
        return _parse_expires_at(payload)
    if isinstance(raw, int | float):
        # Pi-style files commonly store epoch milliseconds.
        seconds = float(raw) / 1000 if raw > 10_000_000_000 else float(raw)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(raw, str):
        text = raw.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    return None


def _expires_to_millis(expires_at: datetime | None) -> int | None:
    if expires_at is None:
        return None
    return int(expires_at.timestamp() * 1000)


def _oauth_credential_from_token_response(
    data: dict[str, Any],
    *,
    provider: str,
    fallback_refresh: str | None = None,
) -> OAuthCredential:
    access = data.get("access_token") or data.get("access")
    if not isinstance(access, str) or not access:
        raise ValueError("OAuth token response has no access_token")
    refresh = data.get("refresh_token") or data.get("refresh") or fallback_refresh
    expires_at = _parse_oauth_expires(data)
    account_id = (
        data.get("accountId") or data.get("account_id") or extract_openai_account_id(access)
    )
    return OAuthCredential(
        provider=provider,
        access=access,
        refresh=refresh if isinstance(refresh, str) else None,
        expires_at=expires_at,
        account_id=account_id if isinstance(account_id, str) else None,
    )


def _response_json(response: Any) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _response_error_detail(response: Any) -> str:
    data = _response_json(response)
    if data:
        error = data.get("error")
        message = data.get("error_description") or data.get("message") or data.get("detail")
        if error and message:
            return f"{response.status_code} {error}: {message}"
        if error:
            return f"{response.status_code} {error}"
        return f"{response.status_code} {data}"
    text = getattr(response, "text", "")
    if isinstance(text, str) and text.strip():
        return f"{response.status_code} {text.strip()[:500]}"
    return f"HTTP {response.status_code}"


def _pkce_verifier() -> str:
    return secrets.token_urlsafe(48).rstrip("=")


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def _parse_expires_at(payload: dict[str, Any]) -> datetime | None:
    if "expires_at" in payload and payload["expires_at"] is not None:
        raw = payload["expires_at"]
        if isinstance(raw, int | float):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        if isinstance(raw, str):
            text = raw.strip()
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            return datetime.fromisoformat(text).astimezone(timezone.utc)
    if "expires_in" in payload and payload["expires_in"] is not None:
        return datetime.now(timezone.utc) + timedelta(seconds=float(payload["expires_in"]))
    return None
