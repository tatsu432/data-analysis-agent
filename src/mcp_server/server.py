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

# Suppress plotly import warning from pandas (we use matplotlib, not plotly)
warnings.filterwarnings(
    "ignore",
    message=".*plotly.*",
    category=UserWarning,
)

from fastmcp import FastMCP

from .analysis_tools import analysis_mcp
from .knowledge_tools import (  # noqa: F401 - Import to register tools
    _ensure_index_built,
)
from .settings import get_settings

# Conditionally import Confluence tools if credentials are available
try:
    from .servers.confluence.tools import confluence_mcp

    _confluence_available = True
except (ImportError, ValueError) as e:
    # Confluence tools unavailable if credentials are missing or library not installed
    _confluence_available = False
    confluence_mcp = None
    logging.getLogger(__name__).warning(
        f"Confluence tools not available: {e}. "
        "Set CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN to enable."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# Create main unified MCP server
main_mcp = FastMCP("Data Analysis Agent MCP Server")


async def import_all_servers() -> None:
    """Import all domain-specific MCP servers into the main server."""
    servers_imported = []

    # Always import analysis and knowledge tools (they're on the same FastMCP instance)
    try:
        await main_mcp.import_server(analysis_mcp, "analysis")
        servers_imported.append("analysis")
        logger.info("✅ Successfully imported analysis tools")
    except Exception as e:
        logger.error(f"❌ Failed to import analysis tools: {e}")
        raise

    # Conditionally import Confluence tools if available
    if _confluence_available and confluence_mcp:
        try:
            # Check if Confluence credentials are set
            confluence_url = os.getenv("CONFLUENCE_URL")
            confluence_username = os.getenv("CONFLUENCE_USERNAME")
            confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN")

            if all([confluence_url, confluence_username, confluence_api_token]):
                await main_mcp.import_server(confluence_mcp, "confluence")
                servers_imported.append("confluence")
                logger.info("✅ Successfully imported Confluence tools")
            else:
                logger.warning(
                    "⚠️  Confluence tools not imported: missing credentials. "
                    "Set CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN to enable."
                )
        except Exception as e:
            logger.warning(f"⚠️  Failed to import Confluence tools: {e}")
            # Don't raise - Confluence is optional

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
        _ensure_index_built()

        # Import all domain servers
        await import_all_servers()

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
