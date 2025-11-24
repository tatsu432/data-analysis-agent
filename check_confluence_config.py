"""Quick diagnostic script to check Confluence configuration for unified MCP server."""

import os

from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Confluence Configuration Check (Unified MCP Server)")
print("=" * 60)

# Check environment variables
mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8082/mcp")
confluence_api_url = os.getenv("CONFLUENCE_URL")
confluence_username = os.getenv("CONFLUENCE_USERNAME")
confluence_token = os.getenv("CONFLUENCE_API_TOKEN")

print("\n1. Environment Variables:")
print(f"   MCP_SERVER_URL: {mcp_server_url}")
print(f"   CONFLUENCE_URL: {confluence_api_url or '❌ NOT SET'}")
print(f"   CONFLUENCE_USERNAME: {confluence_username or '❌ NOT SET'}")
print(f"   CONFLUENCE_API_TOKEN: {'✅ SET' if confluence_token else '❌ NOT SET'}")

# Check Confluence credentials
confluence_configured = all([confluence_api_url, confluence_username, confluence_token])
if confluence_configured:
    print("\n✅ Confluence credentials are configured")
    print("   Confluence tools will be available in the unified MCP server")
else:
    print("\n⚠️  Confluence credentials are not fully configured")
    print("   Confluence tools will NOT be available in the unified MCP server")
    missing = []
    if not confluence_api_url:
        missing.append("CONFLUENCE_URL")
    if not confluence_username:
        missing.append("CONFLUENCE_USERNAME")
    if not confluence_token:
        missing.append("CONFLUENCE_API_TOKEN")
    print(f"   Missing: {', '.join(missing)}")
    print("\n   Add these to your .env file to enable Confluence tools:")
    print("   CONFLUENCE_URL=https://yourcompany.atlassian.net")
    print("   CONFLUENCE_USERNAME=your.email@company.com")
    print("   CONFLUENCE_API_TOKEN=your_api_token_here")

# Check if server is accessible
print("\n2. Unified MCP Server Accessibility:")
import urllib.error
import urllib.request

try:
    # Try to connect to the server
    req = urllib.request.Request(mcp_server_url)
    req.add_header("Accept", "text/event-stream")
    with urllib.request.urlopen(req, timeout=5) as response:
        print(f"   ✅ Server is accessible at {mcp_server_url}")
        print(f"   Status: {response.status}")
except urllib.error.URLError as e:
    print(f"   ❌ Cannot connect to server at {mcp_server_url}")
    print(f"   Error: {e}")
    print("\n   Make sure the unified MCP server is running:")
    print("   python -m src.mcp_server")
except Exception as e:
    print(f"   ⚠️  Unexpected error: {e}")

print("\n" + "=" * 60)
print("Next Steps:")
if not confluence_configured:
    print("1. Add Confluence credentials to .env to enable Confluence tools")
print("2. If server is not accessible, start it:")
print("   python -m src.mcp_server")
print("3. Restart the LangGraph server to reload tools")
print("=" * 60)
