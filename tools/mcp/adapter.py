"""
MCP (Model Context Protocol) adapter for the agent harness.

Connects to any MCP-compatible server and registers its tools into
the harness's ToolRegistry, making them available to agents.

Supports stdio and SSE transports.

Install:
    pip install -e ".[mcp]"

Usage (context manager — recommended):

    from mcp import StdioServerParameters
    from tools.mcp import MCPServerConnection

    params = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])

    async with MCPServerConnection(params, server_name="filesystem") as conn:
        conn.register_tools(tool_registry)
        result = await runtime.run("list files in /tmp")

Usage (manual lifecycle):

    from tools.mcp import register_mcp_server

    conn = await register_mcp_server(tool_registry, params)
    try:
        result = await runtime.run("list files in /tmp")
    finally:
        await conn.close()
"""
from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

logger = logging.getLogger(__name__)


# ── Tool Adapter ──────────────────────────────────────────────────────────────

class MCPToolAdapter:
    """
    Wraps a single MCP tool as a harness-compatible tool.

    Satisfies the harness tool contract:
      - ``name``  attribute (str)
      - ``async execute(**kwargs)`` method
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict,
        session: Any,           # mcp.client.session.ClientSession
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._session = session

    async def execute(self, **kwargs: Any) -> Any:
        """Call the MCP tool and return extracted content."""
        result = await self._session.call_tool(self.name, arguments=kwargs)
        return _extract_content(result)

    def __repr__(self) -> str:
        return f"MCPToolAdapter(name={self.name!r})"


# ── Server Connection ────────────────────────────────────────────────────────

class MCPServerConnection:
    """
    Manages the lifecycle of a connection to an MCP server.

    Connects, discovers tools, and keeps the transport alive so
    agents can invoke tools during a run. Use as an async context
    manager to ensure cleanup::

        async with MCPServerConnection(server_params) as conn:
            conn.register_tools(tool_registry)
            # tools are usable while the context is open
    """

    def __init__(
        self,
        server_params: Any,
        *,
        server_name: str | None = None,
    ) -> None:
        self._params = server_params
        self.server_name = server_name or "mcp-server"
        self._session: Any = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[MCPToolAdapter] = []

    @property
    def tools(self) -> list[MCPToolAdapter]:
        """Discovered tool adapters (empty before connect())."""
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        """Names of discovered tools."""
        return [t.name for t in self._tools]

    async def connect(self) -> list[MCPToolAdapter]:
        """Connect to the MCP server and discover tools."""
        try:
            from mcp import StdioServerParameters  # noqa: F811
            from mcp.client.session import ClientSession
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise ImportError(
                "mcp package not installed. Run: pip install -e \".[mcp]\""
            ) from e

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        try:
            if isinstance(self._params, StdioServerParameters):
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(self._params)
                )
            elif isinstance(self._params, str):
                # SSE transport: params is a URL string
                from mcp.client.sse import sse_client
                read, write = await self._exit_stack.enter_async_context(
                    sse_client(self._params)
                )
            elif isinstance(self._params, dict) and "url" in self._params:
                # SSE transport: params as dict with url + optional headers
                from mcp.client.sse import sse_client
                read, write = await self._exit_stack.enter_async_context(
                    sse_client(**self._params)
                )
            else:
                raise TypeError(
                    f"Unsupported server_params type: {type(self._params)}. "
                    "Use StdioServerParameters, an SSE URL string, or "
                    "a dict with 'url' key."
                )

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()

            result = await self._session.list_tools()
            self._tools = [
                MCPToolAdapter(
                    name=tool.name,
                    description=getattr(tool, "description", None) or "",
                    input_schema=(
                        tool.inputSchema
                        if hasattr(tool, "inputSchema")
                        else {}
                    ),
                    session=self._session,
                )
                for tool in result.tools
            ]

            logger.info(
                "Connected to MCP server %r: %d tools discovered: %s",
                self.server_name, len(self._tools), self.tool_names,
            )
            return self._tools

        except Exception:
            await self._exit_stack.aclose()
            self._exit_stack = None
            raise

    def register_tools(self, tool_registry: Any) -> list[str]:
        """Register all discovered tools into a ToolRegistry. Returns names."""
        for tool in self._tools:
            tool_registry.register(tool)
        return self.tool_names

    async def close(self) -> None:
        """Close the MCP server connection and clean up resources."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._tools = []
            logger.info("Disconnected from MCP server %r", self.server_name)

    async def __aenter__(self) -> MCPServerConnection:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ── Convenience ───────────────────────────────────────────────────────────────

async def register_mcp_server(
    tool_registry: Any,
    server_params: Any,
    *,
    server_name: str | None = None,
) -> MCPServerConnection:
    """
    Convenience: connect to an MCP server and register all its tools.

    Returns the MCPServerConnection — **caller must call conn.close()
    when done**. For automatic cleanup, prefer the context-manager
    pattern with MCPServerConnection directly.
    """
    conn = MCPServerConnection(server_params, server_name=server_name)
    await conn.connect()
    conn.register_tools(tool_registry)
    return conn


# ── Content Extraction ────────────────────────────────────────────────────────

def _extract_content(result: Any) -> Any:
    """
    Extract usable content from an MCP CallToolResult.

    MCP results carry a list of content items (text, images, etc).
    This normalises them into plain Python values for the agent.
    """
    # Error results
    if getattr(result, "isError", False):
        texts = []
        for item in getattr(result, "content", []):
            if hasattr(item, "text"):
                texts.append(item.text)
        error_msg = "\n".join(texts) if texts else "MCP tool returned an error"
        return {"error": error_msg}

    contents = getattr(result, "content", [])
    if not contents:
        return {"result": "no content returned"}

    # Single text content → return as string (or parsed JSON)
    if len(contents) == 1 and hasattr(contents[0], "text"):
        text = contents[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    # Multiple content items → structured list
    parts = []
    for item in contents:
        if hasattr(item, "text"):
            parts.append({"type": "text", "content": item.text})
        elif hasattr(item, "data"):
            parts.append({
                "type": getattr(item, "type", "binary"),
                "size": len(item.data),
            })
        else:
            parts.append({"type": "unknown"})
    return parts
