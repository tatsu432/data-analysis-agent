"""Main MCP server entry point.

This server combines all domain-specific tool modules:
- Analysis tools (data analysis, dataset operations)
- Knowledge tools (knowledge base search, term definitions)
- Confluence tools (page search, create, update, read)
"""

import asyncio
import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from starlette.responses import JSONResponse

from .servers.analysis.server import analysis_mcp
from .servers.knowledge.infrastructure.knowledge_index_manager import ensure_index_built
from .servers.knowledge.server import knowledge_mcp

# Suppress plotly import warning from pandas (we use matplotlib, not plotly)
warnings.filterwarnings(
    "ignore",
    message=".*plotly.*",
    category=UserWarning,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# Create main unified MCP server
main_mcp = FastMCP("Data Analysis Agent MCP Server")


def _create_confluence_stub_server() -> FastMCP:
    """
    Create a stub Confluence MCP server used when ENABLE_CONFLUENCE is not set.

    The tools are registered under the same names as the real Confluence tools
    but always return a deterministic "disabled" response instead of attempting
    any external integration.
    """
    stub = FastMCP("Confluence Tools (disabled)")

    @stub.tool()
    def confluence_search_pages(
        query: str,
        space_key: str | None = None,
        limit: int = 10,
    ) -> dict:
        return {
            "error": "Confluence integration is disabled in this environment.",
            "enabled": False,
            "operation": "search_pages",
            "query": query,
            "space_key": space_key,
            "limit": limit,
        }

    @stub.tool()
    def confluence_get_page(page_id: str) -> dict:
        return {
            "error": "Confluence integration is disabled in this environment.",
            "enabled": False,
            "operation": "get_page",
            "page_id": page_id,
        }

    @stub.tool()
    def confluence_create_page(
        space_key: str,
        title: str,
        body: str,
        parent_id: str | None = None,
    ) -> dict:
        return {
            "error": "Confluence integration is disabled in this environment.",
            "enabled": False,
            "operation": "create_page",
            "space_key": space_key,
            "title": title,
            "parent_id": parent_id,
        }

    @stub.tool()
    def confluence_update_page(
        page_id: str,
        title: str | None = None,
        body: str | None = None,
    ) -> dict:
        return {
            "error": "Confluence integration is disabled in this environment.",
            "enabled": False,
            "operation": "update_page",
            "page_id": page_id,
        }

    return stub


@main_mcp.custom_route("/health", methods=["GET"])
async def health_check(request) -> JSONResponse:  # type: ignore[override]
    """HTTP health endpoint used by CI and external monitors."""
    return JSONResponse({"status": "ok", "service": "mcp-server"})


async def import_all_servers() -> None:
    """Import all domain-specific MCP servers into the main server."""
    servers_imported = []

    # Always import analysis tools
    try:
        await main_mcp.import_server(analysis_mcp, "analysis")
        servers_imported.append("analysis")
        logger.info("✅ Successfully imported analysis tools")
    except Exception as e:
        logger.error(f"❌ Failed to import analysis tools: {e}")
        raise

    # Always import knowledge tools
    try:
        await main_mcp.import_server(knowledge_mcp, "knowledge")
        servers_imported.append("knowledge")
        logger.info("✅ Successfully imported knowledge tools")
    except Exception as e:
        logger.error(f"❌ Failed to import knowledge tools: {e}")
        raise

    # Confluence tools: either real or stub based on ENABLE_CONFLUENCE
    enable_confluence = os.getenv("ENABLE_CONFLUENCE", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if not enable_confluence:
        # Always register a stub server so calls fail deterministically in CI.
        try:
            stub_server = _create_confluence_stub_server()
            await main_mcp.import_server(stub_server, "confluence")
            servers_imported.append("confluence-stub")
            logger.info(
                "ℹ️  Confluence tools disabled (ENABLE_CONFLUENCE not set); using stub tools."
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to import Confluence stub tools: {e}")
    else:
        # ENABLE_CONFLUENCE=true: require credentials and real tools; fail fast if misconfigured.
        try:
            from .servers.confluence.server import confluence_mcp
        except Exception as e:
            logger.error(
                "❌ ENABLE_CONFLUENCE is true but Confluence server could not be imported: %s",
                e,
            )
            raise

        confluence_url = os.getenv("CONFLUENCE_URL")
        confluence_username = os.getenv("CONFLUENCE_USERNAME")
        confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN")

        if not all([confluence_url, confluence_username, confluence_api_token]):
            raise RuntimeError(
                "ENABLE_CONFLUENCE=true but Confluence credentials are missing. "
                "Set CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN."
            )

        try:
            await main_mcp.import_server(confluence_mcp, "confluence")
            servers_imported.append("confluence")
            logger.info("✅ Successfully imported Confluence tools")
        except Exception as e:
            logger.error(f"❌ Failed to import Confluence tools: {e}")
            raise

    logger.info(
        f"📦 Successfully configured unified MCP server with {len(servers_imported)} domain(s): {', '.join(servers_imported)}"
    )


async def main() -> None:
    """Main function."""
    try:
        logger.info("=" * 60)
        logger.info("Starting unified MCP server...")
        logger.info("=" * 60)

        # Ensure knowledge index is built on startup
        ensure_index_built()

        # Import all domain servers
        await import_all_servers()

        from .settings import get_settings

        settings = get_settings()
        logger.info(f"Server will run on: http://{settings.host}:{settings.port}/mcp")
        logger.info("=" * 60)

        # Run the unified server
        await main_mcp.run_async(
            transport=settings.transport, host=settings.host, port=settings.port
        )
    except KeyboardInterrupt:
        logger.info("\n👋 MCP Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to run MCP server: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
