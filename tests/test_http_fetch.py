"""HTTPFetch tests against a mocked httpx transport (no network)."""
from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx")
respx = pytest.importorskip("respx")

from tools.builtin.http_fetch import HTTPFetch  # noqa: E402


@respx.mock
async def test_http_fetch_returns_body_and_status():
    respx.get("https://example.com/x").mock(
        return_value=httpx.Response(
            200, text='{"ok": true}',
            headers={"content-type": "application/json"},
        ),
    )
    tool = HTTPFetch()
    out = await tool.execute(url="https://example.com/x")
    assert out["status"] == 200
    assert out["body"] == '{"ok": true}'
    assert out["content_type"] == "application/json"
    assert out["truncated"] is False


@respx.mock
async def test_http_fetch_truncates_long_body():
    big = "a" * 200_000
    respx.get("https://example.com/big").mock(
        return_value=httpx.Response(200, text=big),
    )
    tool = HTTPFetch(max_bytes=1024)
    out = await tool.execute(url="https://example.com/big")
    assert out["truncated"] is True
    assert len(out["body"]) == 1024


@respx.mock
async def test_http_fetch_returns_error_on_network_failure():
    respx.get("https://example.com/err").mock(
        side_effect=httpx.ConnectError("boom"),
    )
    tool = HTTPFetch()
    out = await tool.execute(url="https://example.com/err")
    assert "error" in out
    assert "ConnectError" in out["error"]


@respx.mock
async def test_http_fetch_follows_redirects():
    respx.get("https://example.com/redirect").mock(
        return_value=httpx.Response(302, headers={"location": "https://example.com/final"}),
    )
    respx.get("https://example.com/final").mock(
        return_value=httpx.Response(200, text="landed"),
    )
    tool = HTTPFetch()
    out = await tool.execute(url="https://example.com/redirect")
    assert out["status"] == 200
    assert out["body"] == "landed"
    assert out["url"].endswith("/final")
