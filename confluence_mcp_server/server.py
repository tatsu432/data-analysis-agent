"""Confluence MCP server entry point."""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from confluence_tools import confluence_mcp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables from project root
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()


async def main() -> None:
    """Main function."""
    try:
        # Validate required environment variables
        required_vars = ["CONFLUENCE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please set them in your .env file or environment."
            )
        
        url = os.getenv("CONFLUENCE_URL")
        port = int(os.getenv("CONFLUENCE_MCP_PORT", "8083"))
        
        logger.info("=" * 60)
        logger.info("Starting Confluence MCP Server...")
        logger.info(f"Confluence URL: {url}")
        logger.info(f"Server will run on: http://0.0.0.0:{port}/mcp")
        logger.info("=" * 60)
        
        # Run the MCP server
        # Use "streamable-http" to match what the LangGraph tool loader expects
        await confluence_mcp.run_async(
            transport="streamable-http",
            host="0.0.0.0",
            port=port,
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Confluence MCP Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to run Confluence MCP server: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())

