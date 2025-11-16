"""MCP tool loader for LangGraph agent."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing_extensions import TypedDict

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ServerConnections(TypedDict):
    """Type definition for server connections."""

    data_analysis: StreamableHttpConnection


class MCPToolLoader:
    """Class-based loader for MCP tools from the data analysis server."""

    def __init__(self, max_in_flight: int = 3) -> None:
        """Initialize the MCP tool loader."""
        self.max_in_flight = max_in_flight
        self._connections = self._build_connections()

    def get_default_headers(self, token: str | None = None) -> dict[str, str]:
        """Standard SSE/streaming-friendly headers (+ optional bearer + X-Request-ID)."""
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": str(uuid.uuid4()),  # unique correlation ID per request
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _build_connections(self) -> ServerConnections:
        """Build connection dicts using ONLY supported kwargs."""
        data_analysis_conn: StreamableHttpConnection = {
            "transport": "streamable_http",
            "url": settings.data_analysis_mcp_server_url,
            "headers": self.get_default_headers(),
            "timeout": timedelta(seconds=settings.mcp_server_timeout),
            "sse_read_timeout": timedelta(
                seconds=settings.mcp_server_sse_read_timeout
            ),
            "terminate_on_close": settings.mcp_server_terminate_on_close,
        }

        return {"data_analysis": data_analysis_conn}

    async def _load_one_server(self, name: str, conn: dict) -> list[BaseTool]:
        """Load tools from a single MCP server; raises on hard errors."""
        client = MultiServerMCPClient(connections={name: conn})
        tools = await client.get_tools()
        logger.info("MCP server %s: loaded %d tools", name, len(tools))
        return tools

    async def _load_all_servers(self) -> list[BaseTool]:
        """Load servers independently; collect successes; log failures."""
        sem = asyncio.Semaphore(self.max_in_flight)
        results: list[BaseTool] = []

        async def runner(
            n: str, c: StreamableHttpConnection
        ) -> tuple[str, list[BaseTool], Exception | None]:
            async with sem:
                try:
                    tools = await self._load_one_server(n, c)
                    return (n, tools, None)
                except Exception as e:
                    return (n, [], e)

        batches = await asyncio.gather(
            *(runner(n, c) for n, c in self._connections.items()),
            return_exceptions=False,
        )
        for name, tools, err in batches:
            if err:
                logger.error(
                    "MCP server %s FAILED during initialize/get_tools: %r", name, err
                )
            else:
                results.extend(tools)

        if not results:
            raise RuntimeError("All MCP servers failed to initialize; see logs above.")
        return results

    @asynccontextmanager
    async def get_mcp_tools(self) -> AsyncGenerator[list[BaseTool], None]:
        """
        Yields a combined list of tools from the MCP server.
        Loads the server independently to avoid all-or-nothing failures.
        """
        tools = await self._load_all_servers()
        logger.info(
            "get_mcp_tools yielded %d total tools from %d servers.",
            len(tools),
            len(self._connections),
        )
        try:
            yield tools
        finally:
            pass

