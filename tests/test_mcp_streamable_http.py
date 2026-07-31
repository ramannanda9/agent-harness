"""End-to-end streamable-HTTP transport test against a real in-process MCP server.

The streamable-HTTP path is pure SDK wiring — headers, timeouts and auth are
folded into an ``httpx2.AsyncClient`` that we hand to ``streamable_http_client``.
Mocks would only assert that we call what we think we call, so this starts an
actual server on a loopback port and drives a real connection through it.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from tools.mcp.adapter import MCPServerConnection
from tools.mcp.auth import ApiKeyMCPAuth, StreamableHttpServerParams

pytestmark = pytest.mark.anyio


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _serve(port: int, seen_headers: dict[str, str]) -> asyncio.Task[None]:
    """Start an MCP server exposing one tool; return the serving task."""
    import uvicorn
    from mcp.server import MCPServer

    server = MCPServer(name="test-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    app = server.streamable_http_app()

    class _CaptureHeaders:
        def __init__(self, inner):
            self._inner = inner

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                seen_headers.update({k.decode(): v.decode() for k, v in scope.get("headers", [])})
            await self._inner(scope, receive, send)

    config = uvicorn.Config(
        _CaptureHeaders(app),
        host="127.0.0.1",
        port=port,
        log_level="error",
        lifespan="on",
    )
    uv = uvicorn.Server(config)
    task = asyncio.create_task(uv.serve())

    for _ in range(200):
        if uv.started:
            return task
        await asyncio.sleep(0.05)
    task.cancel()
    raise RuntimeError("server did not start")


async def test_streamable_http_connect_discovers_and_calls_tool(monkeypatch):
    port = _free_port()
    seen_headers: dict[str, str] = {}
    task = await _serve(port, seen_headers)

    try:
        monkeypatch.setenv("TEST_MCP_KEY", "secret-value")
        params = StreamableHttpServerParams(
            url=f"http://127.0.0.1:{port}/mcp",
            headers={"X-Custom": "from-params"},
            timeout=10.0,
            sse_read_timeout=30.0,
        )
        auth = ApiKeyMCPAuth({"X-Api-Key": "TEST_MCP_KEY"})

        async with MCPServerConnection(params, auth=auth) as conn:
            assert "add" in conn.tool_names
            tool = next(t for t in conn.tools if t.name == "add")
            assert await tool.execute(a=2, b=3) == 5
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # Both the params headers and the resolved auth headers must reach the wire.
    assert seen_headers.get("x-custom") == "from-params"
    assert seen_headers.get("x-api-key") == "secret-value"
