"""Quick diagnostic script to check Confluence MCP configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Confluence MCP Configuration Check")
print("=" * 60)

# Check environment variables
confluence_url = os.getenv("CONFLUENCE_MCP_SERVER_URL")
confluence_api_url = os.getenv("CONFLUENCE_URL")
confluence_username = os.getenv("CONFLUENCE_USERNAME")
confluence_token = os.getenv("CONFLUENCE_API_TOKEN")

print("\n1. Environment Variables:")
print(f"   CONFLUENCE_MCP_SERVER_URL: {confluence_url or '❌ NOT SET'}")
print(f"   CONFLUENCE_URL: {confluence_api_url or '❌ NOT SET'}")
print(f"   CONFLUENCE_USERNAME: {confluence_username or '❌ NOT SET'}")
print(f"   CONFLUENCE_API_TOKEN: {'✅ SET' if confluence_token else '❌ NOT SET'}")

if not confluence_url:
    print("\n❌ CONFLUENCE_MCP_SERVER_URL is not set!")
    print("   Add this to your .env file:")
    print("   CONFLUENCE_MCP_SERVER_URL=http://localhost:8083/mcp")
else:
    print(f"\n✅ CONFLUENCE_MCP_SERVER_URL is set to: {confluence_url}")

# Check if server is accessible
if confluence_url:
    print("\n2. Server Accessibility:")
    import urllib.request
    import urllib.error
    
    try:
        # Try to connect to the server
        req = urllib.request.Request(confluence_url)
        req.add_header("Accept", "text/event-stream")
        with urllib.request.urlopen(req, timeout=5) as response:
            print(f"   ✅ Server is accessible at {confluence_url}")
            print(f"   Status: {response.status}")
    except urllib.error.URLError as e:
        print(f"   ❌ Cannot connect to server at {confluence_url}")
        print(f"   Error: {e}")
        print("\n   Make sure the Confluence MCP server is running:")
        print("   python confluence_mcp_server/server.py")
    except Exception as e:
        print(f"   ⚠️  Unexpected error: {e}")

print("\n" + "=" * 60)
print("Next Steps:")
print("1. If CONFLUENCE_MCP_SERVER_URL is not set, add it to .env")
print("2. If server is not accessible, start it:")
print("   python confluence_mcp_server/server.py")
print("3. Restart the LangGraph server to reload tools")
print("=" * 60)

