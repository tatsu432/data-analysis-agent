"""MCP tool loader for LangGraph agent."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from typing_extensions import TypedDict

from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ServerConnections(TypedDict, total=False):
    """Type definition for server connections."""

    data_analysis: StreamableHttpConnection
    confluence: StreamableHttpConnection


class MCPToolLoader:
    """Class-based loader for MCP tools from multiple servers (data analysis and optionally Confluence)."""

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
        connections: ServerConnections = {}

        # Always include data analysis server
        data_analysis_conn: StreamableHttpConnection = {
            "transport": "streamable_http",
            "url": settings.data_analysis_mcp_server_url,
            "headers": self.get_default_headers(),
            "timeout": timedelta(seconds=settings.mcp_server_timeout),
            "sse_read_timeout": timedelta(seconds=settings.mcp_server_sse_read_timeout),
            "terminate_on_close": settings.mcp_server_terminate_on_close,
        }
        connections["data_analysis"] = data_analysis_conn
        logger.info(
            "Data analysis MCP server configured: %s",
            settings.data_analysis_mcp_server_url,
        )

        # Conditionally include Confluence server if URL is configured
        if settings.confluence_mcp_server_url:
            confluence_conn: StreamableHttpConnection = {
                "transport": "streamable_http",
                "url": settings.confluence_mcp_server_url,
                "headers": self.get_default_headers(),
                "timeout": timedelta(seconds=settings.mcp_server_timeout),
                "sse_read_timeout": timedelta(
                    seconds=settings.mcp_server_sse_read_timeout
                ),
                "terminate_on_close": settings.mcp_server_terminate_on_close,
            }
            connections["confluence"] = confluence_conn
            logger.info(
                "✅ Confluence MCP server configured: %s",
                settings.confluence_mcp_server_url,
            )
        else:
            logger.warning(
                "⚠️  Confluence MCP server not configured (CONFLUENCE_MCP_SERVER_URL not set)"
            )

        logger.info(
            "Building connections for %d MCP server(s): %s",
            len(connections),
            list(connections.keys()),
        )
        return connections

    async def _load_one_server(self, name: str, conn: dict) -> list[BaseTool]:
        """Load tools from a single MCP server; raises on hard errors."""
        logger.info(
            "Connecting to MCP server: %s at %s", name, conn.get("url", "unknown")
        )
        try:
            client = MultiServerMCPClient(connections={name: conn})
            tools = await client.get_tools()
            tool_names = [tool.name for tool in tools]
            logger.info(
                "MCP server %s: loaded %d tools: %s", name, len(tools), tool_names
            )
            return tools
        except Exception as e:
            logger.error(
                "Failed to load tools from MCP server %s: %s", name, e, exc_info=True
            )
            raise

    async def _load_all_servers(self) -> list[BaseTool]:
        """Load servers independently; collect successes; log failures."""
        logger.info("=" * 60)
        logger.info("Loading tools from %d MCP server(s)...", len(self._connections))
        logger.info("=" * 60)

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
                    logger.error(
                        "❌ Failed to load tools from server '%s': %s",
                        n,
                        e,
                        exc_info=True,
                    )
                    return (n, [], e)

        batches = await asyncio.gather(
            *(runner(n, c) for n, c in self._connections.items()),
            return_exceptions=False,
        )

        for name, tools, err in batches:
            if err:
                logger.error(
                    "❌ MCP server '%s' FAILED during initialize/get_tools: %r",
                    name,
                    err,
                    exc_info=True,
                )
                if name == "confluence":
                    logger.warning(
                        "⚠️  Confluence tools will not be available. "
                        "Check that the Confluence MCP server is running at: %s",
                        self._connections.get("confluence", {}).get("url", "unknown"),
                    )
            else:
                logger.info(
                    "✅ Successfully loaded %d tools from server: %s", len(tools), name
                )
                results.extend(tools)

        # Log summary
        all_tool_names = [tool.name for tool in results]
        logger.info("=" * 60)
        logger.info("📦 Total tools loaded: %d", len(results))
        logger.info("Tool names: %s", all_tool_names)

        # Check for Confluence tools
        confluence_tools = [
            name for name in all_tool_names if "confluence" in name.lower()
        ]
        if confluence_tools:
            logger.info("✅ Confluence tools found: %s", confluence_tools)
        else:
            logger.warning(
                "⚠️  No Confluence tools found. "
                "If you expected Confluence tools, check:"
                " 1. CONFLUENCE_MCP_SERVER_URL is set in .env"
                " 2. Confluence MCP server is running"
                " 3. Server logs for connection errors"
            )
        logger.info("=" * 60)

        if not results:
            # Only raise if data_analysis server failed (required)
            # Confluence server is optional
            data_analysis_failed = any(
                name == "data_analysis" and err is not None for name, _, err in batches
            )
            if data_analysis_failed:
                raise RuntimeError(
                    "Required MCP server (data_analysis) failed to initialize; see logs above."
                )
            # If only optional servers failed, log warning but continue
            logger.warning(
                "No tools loaded from any MCP server, but data_analysis server succeeded. "
                "This may indicate a configuration issue."
            )
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
