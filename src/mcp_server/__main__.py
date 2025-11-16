"""Entry point for running the MCP server as a module.

Usage:
    python -m src.mcp_server
"""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.mcp_server.server import main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("🚀 Starting MCP Server with data analysis tools...")
    print("=" * 60)
    print("This server provides the tools that the LangGraph agent needs.")
    print("Keep this running while using the agent.")
    print("=" * 60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 MCP Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start MCP server: {e}")
        sys.exit(1)

