"""Tests for the localhost OAuth callback server + browser-open helper."""

from __future__ import annotations

import asyncio
import urllib.request
from unittest.mock import patch

import pytest

from harness.oauth_browser import bind_callback_server, open_or_print_url, wait_for_oauth_callback

# ── open_or_print_url ─────────────────────────────────────────────────────────


def test_open_or_print_url_always_prints(capsys):
    with patch("webbrowser.open", return_value=True) as opener:
        open_or_print_url("https://example.com/auth", prefix="Open:")
    captured = capsys.readouterr()
    assert "https://example.com/auth" in captured.out
    opener.assert_called_once()


def test_open_or_print_url_survives_webbrowser_failure(capsys):
    with patch("webbrowser.open", side_effect=RuntimeError("no display")):
        # Must not raise — printing the URL is the fallback.
        open_or_print_url("https://example.com/auth")
    assert "https://example.com/auth" in capsys.readouterr().out


# ── wait_for_oauth_callback ───────────────────────────────────────────────────


async def test_callback_returns_code_and_state():
    server, port, future = bind_callback_server(port=0, path="/cb")
    try:

        def hit() -> None:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/cb?code=AUTHCODE&state=ST").read()

        await asyncio.get_running_loop().run_in_executor(None, hit)
        code, state = await asyncio.wait_for(future, timeout=2.0)
    finally:
        server.shutdown()
    assert code == "AUTHCODE"
    assert state == "ST"


async def test_callback_returns_code_with_no_state():
    server, port, future = bind_callback_server(port=0, path="/cb")
    try:

        def hit() -> None:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/cb?code=X").read()

        await asyncio.get_running_loop().run_in_executor(None, hit)
        code, state = await asyncio.wait_for(future, timeout=2.0)
    finally:
        server.shutdown()
    assert code == "X"
    assert state is None


async def test_callback_raises_on_oauth_error():
    server, port, future = bind_callback_server(port=0, path="/cb")
    try:

        def hit() -> None:
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/cb?error=access_denied&error_description=user+declined"
                ).read()
            except Exception:  # noqa: BLE001 — 400 raises HTTPError, fine
                pass

        await asyncio.get_running_loop().run_in_executor(None, hit)
        with pytest.raises(RuntimeError, match="access_denied"):
            await asyncio.wait_for(future, timeout=2.0)
    finally:
        server.shutdown()


async def test_callback_ignores_unrelated_paths():
    """Hits to /favicon.ico, /robots.txt etc. shouldn't resolve the future."""
    server, port, future = bind_callback_server(port=0, path="/cb")
    try:

        def hit_noise() -> None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/favicon.ico").read()
            except Exception:  # noqa: BLE001 — 404 raises HTTPError, fine
                pass

        await asyncio.get_running_loop().run_in_executor(None, hit_noise)
        # future must still be pending
        await asyncio.sleep(0.05)
        assert not future.done()

        def hit_real() -> None:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/cb?code=OK").read()

        await asyncio.get_running_loop().run_in_executor(None, hit_real)
        code, _ = await asyncio.wait_for(future, timeout=2.0)
        assert code == "OK"
    finally:
        server.shutdown()


async def test_wait_for_oauth_callback_times_out():
    with pytest.raises(asyncio.TimeoutError):
        await wait_for_oauth_callback(port=0, timeout=0.1)
