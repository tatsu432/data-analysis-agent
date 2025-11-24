"""Test script to verify unified MCP server includes Confluence tools."""

import asyncio
import os
from datetime import timedelta

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

load_dotenv()


async def test_confluence_tools():
    """Test if unified MCP server includes Confluence tools."""

    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8082/mcp")

    print(f"Testing unified MCP server at: {mcp_server_url}")
    print("=" * 60)

    # Create connection
    connection: StreamableHttpConnection = {
        "transport": "streamable_http",
        "url": mcp_server_url,
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
        client = MultiServerMCPClient(connections={"unified": connection})

        # Get tools
        print("Loading tools from unified MCP server...")
        tools = await client.get_tools()

        print("\n✅ Successfully connected to unified MCP server!")
        print(f"📦 Loaded {len(tools)} tool(s) total\n")

        # Separate tools by category
        confluence_tools = [t for t in tools if "confluence" in t.name.lower()]
        analysis_tools = [
            t
            for t in tools
            if any(x in t.name.lower() for x in ["dataset", "analysis", "schema"])
        ]
        knowledge_tools = [
            t
            for t in tools
            if any(x in t.name.lower() for x in ["knowledge", "term", "document"])
        ]

        print("Tool Categories:")
        print(f"  - Analysis tools: {len(analysis_tools)}")
        print(f"  - Knowledge tools: {len(knowledge_tools)}")
        print(f"  - Confluence tools: {len(confluence_tools)}")

        if confluence_tools:
            print("\n✅ Confluence tools found:")
            for tool in confluence_tools:
                print(f"  - {tool.name}")
                if hasattr(tool, "description"):
                    desc = (
                        tool.description[:80] + "..."
                        if len(tool.description) > 80
                        else tool.description
                    )
                    print(f"    {desc}")
        else:
            print("\n⚠️  No Confluence tools found")
            print(
                "   Make sure CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN are set"
            )

        # Check for expected Confluence tools
        tool_names = [tool.name.lower() for tool in tools]
        expected_tools = [
            "confluence_search_pages",
            "confluence_get_page",
            "confluence_create_page",
        ]

        print("\n" + "=" * 60)
        print("Checking for expected Confluence tools:")
        for expected in expected_tools:
            found = any(expected.lower() in name for name in tool_names)
            status = "✅" if found else "❌"
            print(f"  {status} {expected}")

    except Exception as e:
        print(f"\n❌ Error connecting to unified MCP server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the unified MCP server is running:")
        print("   python -m src.mcp_server")
        print("2. Check that MCP_SERVER_URL is correct in .env")
        print("3. Verify the server is accessible:")
        print(f"   curl {mcp_server_url}")
        raise


if __name__ == "__main__":
    asyncio.run(test_confluence_tools())
