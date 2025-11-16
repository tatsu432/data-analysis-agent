"""Main MCP server entry point."""

import asyncio
import logging

from .analysis_tools import analysis_mcp
from .settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def main() -> None:
    """Main function."""
    try:
        logger.info("Starting MCP server with data analysis tools...")
        await analysis_mcp.run_async(
            transport=settings.transport, host=settings.host, port=settings.port
        )
    except Exception as e:
        logger.error(f"Failed to run MCP: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
