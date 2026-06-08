from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from harness.llm.auth import (
    AnthropicClaudeCodeOAuthClient,
    AuthFileOAuthProvider,
    OAuthCredential,
    OpenAICodexOAuthClient,
    default_auth_file,
)
from harness.tool_policy import ToolPolicyStore, default_policy_file

PROVIDERS = ["openai-codex", "claude-code"]


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent-harness", description="agent-harness utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    login = sub.add_parser("login", help="log in to a provider")
    login.add_argument("provider", choices=PROVIDERS)
    login.add_argument("--auth-file", default=str(default_auth_file()))

    status = sub.add_parser("auth", help="inspect or clear provider auth")
    status_sub = status.add_subparsers(dest="auth_command", required=True)
    status_cmd = status_sub.add_parser("status", help="show auth status")
    status_cmd.add_argument("provider", choices=PROVIDERS)
    status_cmd.add_argument("--auth-file", default=str(default_auth_file()))
    logout_cmd = status_sub.add_parser("logout", help="remove auth credentials")
    logout_cmd.add_argument("provider", choices=PROVIDERS)
    logout_cmd.add_argument("--auth-file", default=str(default_auth_file()))

    policy = sub.add_parser("policy", help="manage persistent tool policy")
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)
    policy_list = policy_sub.add_parser("list", help="list persistent policy rules")
    policy_list.add_argument("--policy-file", default=str(default_policy_file()))
    policy_revoke = policy_sub.add_parser("revoke", help="remove one policy rule")
    policy_revoke.add_argument("rule_id")
    policy_revoke.add_argument("--policy-file", default=str(default_policy_file()))
    policy_clear = policy_sub.add_parser("clear", help="remove all policy rules")
    policy_clear.add_argument("--policy-file", default=str(default_policy_file()))

    trace = sub.add_parser("trace", help="view or replay a recorded run trace")
    trace_sub = trace.add_subparsers(dest="trace_command", required=True)
    trace_view = trace_sub.add_parser("view", help="open a local web viewer for a trace")
    trace_view.add_argument("path", help="path to a JSONL trace produced by record_trace")
    trace_view.add_argument("--port", type=int, default=8765)
    trace_view.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    trace_replay = trace_sub.add_parser("replay", help="dump a trace to stdout via ConsoleRenderer")
    trace_replay.add_argument("path", help="path to a JSONL trace produced by record_trace")
    trace_replay.add_argument(
        "--realtime", action="store_true", help="preserve recorded inter-event timing"
    )
    trace_replay.add_argument("--speed", type=float, default=1.0, help="realtime speed multiplier")

    args = parser.parse_args()
    try:
        if args.command == "login":
            if args.provider == "openai-codex":
                return asyncio.run(_login_openai_codex(Path(args.auth_file).expanduser()))
            if args.provider == "claude-code":
                return asyncio.run(_login_claude_code(Path(args.auth_file).expanduser()))
        if args.command == "auth" and args.auth_command == "status":
            if args.provider == "openai-codex":
                return _status_oauth_provider(Path(args.auth_file).expanduser(), "openai-codex")
            if args.provider == "claude-code":
                return _status_oauth_provider(Path(args.auth_file).expanduser(), "claude-code")
        if args.command == "auth" and args.auth_command == "logout":
            if args.provider == "openai-codex":
                return _logout_oauth_provider(Path(args.auth_file).expanduser(), "openai-codex")
            if args.provider == "claude-code":
                return _logout_oauth_provider(Path(args.auth_file).expanduser(), "claude-code")
        if args.command == "policy":
            path = Path(args.policy_file).expanduser()
            if args.policy_command == "list":
                return _policy_list(path)
            if args.policy_command == "revoke":
                return _policy_revoke(path, args.rule_id)
            if args.policy_command == "clear":
                return _policy_clear(path)
        if args.command == "trace":
            if args.trace_command == "view":
                from harness.trace_viewer import serve

                serve(args.path, port=args.port, open_browser=not args.no_open)
                return 0
            if args.trace_command == "replay":
                return asyncio.run(
                    _trace_replay(args.path, realtime=args.realtime, speed=args.speed)
                )
    except Exception as e:
        print(f"agent-harness: {e}", file=sys.stderr)
        return 1
    parser.error("unsupported command")
    return 2


async def _login_openai_codex(path: Path) -> int:
    from harness.oauth_browser import open_or_print_url

    client = OpenAICodexOAuthClient()
    try:
        device = await client.request_device_code()
        print("OpenAI Codex login")
        open_or_print_url(device.verification_uri, prefix="Open:")
        print(f"Code: {device.user_code}")
        print("Waiting for authorization...")
        cred = await client.poll_device_code(device)
    finally:
        await client.aclose()
    _write_oauth_credential(path, cred)
    print(f"Logged in to openai-codex. Credentials saved to {path}")
    return 0


async def _login_claude_code(path: Path) -> int:
    from harness.oauth_browser import open_or_print_url

    client = AnthropicClaudeCodeOAuthClient()
    try:
        login = client.begin_login()
        print("Claude Code login")
        # Anthropic owns the redirect URI (console.anthropic.com), so we
        # can't auto-capture the callback here. Best we can do is open the
        # browser for the user and let them paste the result.
        open_or_print_url(login.url, prefix="Open:")
        print("Paste the final callback URL, or the code#state value.")
        callback_input = input("Callback: ")
        cred = await client.finish_login(login, callback_input)
    finally:
        await client.aclose()
    _write_oauth_credential(path, cred)
    print(f"Logged in to claude-code. Credentials saved to {path}")
    return 0


def _status_oauth_provider(path: Path, provider_name: str) -> int:
    provider = AuthFileOAuthProvider(path, provider=provider_name)
    try:
        cred = provider._read_credential()
    except FileNotFoundError:
        print(f"Not logged in: {path} does not exist")
        return 1
    except Exception as e:
        print(f"Not logged in: {e}")
        return 1
    status = {
        "provider": provider_name,
        "auth_file": str(path),
        "account_id": cred.account_id,
        "expires_at": cred.expires_at.isoformat() if cred.expires_at else None,
        "expired": cred.is_expired(),
    }
    print(json.dumps(status, indent=2))
    return 0


def _logout_oauth_provider(path: Path, provider_name: str) -> int:
    provider = AuthFileOAuthProvider(
        path, provider=provider_name, require_private_permissions=False
    )
    provider.clear()
    print(f"Removed {provider_name} credentials from {path}")
    return 0


def _write_oauth_credential(path: Path, cred: OAuthCredential) -> None:
    provider = AuthFileOAuthProvider(
        path, provider=cred.provider, require_private_permissions=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}")
        if os.name != "nt":
            path.chmod(0o600)
    provider._write_credential(cred)


def _policy_list(path: Path) -> int:
    store = ToolPolicyStore(path)
    rules = [rule.to_dict() for rule in store.list_rules()]
    print(json.dumps({"policy_file": str(path), "rules": rules}, indent=2))
    return 0


def _policy_revoke(path: Path, rule_id: str) -> int:
    if not ToolPolicyStore(path).revoke(rule_id):
        print(f"Policy rule not found: {rule_id}", file=sys.stderr)
        return 1
    print(f"Removed policy rule: {rule_id}")
    return 0


def _policy_clear(path: Path) -> int:
    count = ToolPolicyStore(path).clear()
    print(f"Removed {count} policy rule(s)")
    return 0


async def _trace_replay(path: str, *, realtime: bool, speed: float) -> int:
    """Read a JSONL trace and render it via ConsoleRenderer.

    Esc cancels the replay — useful when ``--realtime`` is in effect and
    the original run was long.
    """
    from harness.console import ConsoleRenderer  # noqa: PLC0415
    from harness.trace import replay  # noqa: PLC0415

    renderer = ConsoleRenderer()
    cancelled, _ = await renderer.render_stream(replay(path, realtime=realtime, speed=speed))
    return 130 if cancelled else 0


if __name__ == "__main__":
    raise SystemExit(main())
