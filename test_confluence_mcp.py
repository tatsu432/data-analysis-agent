"""Test script to verify Confluence MCP server is accessible and tools are loaded."""

import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from datetime import timedelta

load_dotenv()


async def test_confluence_mcp():
    """Test if Confluence MCP server is accessible and tools are loaded."""
    
    confluence_url = os.getenv("CONFLUENCE_MCP_SERVER_URL")
    
    if not confluence_url:
        print("❌ CONFLUENCE_MCP_SERVER_URL not set in .env file")
        return
    
    print(f"Testing Confluence MCP server at: {confluence_url}")
    print("=" * 60)
    
    # Create connection
    connection: StreamableHttpConnection = {
        "transport": "streamable_http",
        "url": confluence_url,
        "headers": {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
        "timeout": timedelta(seconds=30),
        "sse_read_timeout": timedelta(seconds=60),
        "terminate_on_close": True,
    }
    
    try:
        # Create client
        client = MultiServerMCPClient(connections={"confluence": connection})
        
        # Get tools
        print("Loading tools from Confluence MCP server...")
        tools = await client.get_tools()
        
        print(f"\n✅ Successfully connected to Confluence MCP server!")
        print(f"📦 Loaded {len(tools)} tool(s):\n")
        
        for tool in tools:
            print(f"  - {tool.name}")
            if hasattr(tool, "description"):
                print(f"    Description: {tool.description[:100]}...")
        
        # Check for expected tools
        tool_names = [tool.name.lower() for tool in tools]
        expected_tools = [
            "confluence_search_pages",
            "confluence_get_page",
            "confluence_create_page",
        ]
        
        print("\n" + "=" * 60)
        print("Checking for expected tools:")
        for expected in expected_tools:
            found = any(expected.lower() in name for name in tool_names)
            status = "✅" if found else "❌"
            print(f"  {status} {expected}")
        
    except Exception as e:
        print(f"\n❌ Error connecting to Confluence MCP server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the Confluence MCP server is running:")
        print("   python confluence_mcp_server/server.py")
        print("2. Check that CONFLUENCE_MCP_SERVER_URL is correct in .env")
        print("3. Verify the server is accessible:")
        print(f"   curl {confluence_url}")
        raise


if __name__ == "__main__":
    asyncio.run(test_confluence_mcp())

