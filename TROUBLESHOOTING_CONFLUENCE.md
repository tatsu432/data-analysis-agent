# Troubleshooting Confluence MCP Integration

## Issue: "Confluence search tool not found"

### Root Cause
The Confluence MCP server was using `transport="http"` but the LangGraph tool loader expects `transport="streamable-http"` for proper SSE (Server-Sent Events) support.

### Fix Applied
Updated `confluence_mcp_server/server.py` to use `transport="streamable-http"` instead of `transport="http"`.

### Verification Steps

1. **Restart the Confluence MCP Server:**
   ```bash
   # Stop the current server (Ctrl+C) and restart:
   python confluence_mcp_server/server.py
   ```

2. **Check the logs when starting the LangGraph server:**
   You should see logs like:
   ```
   Connecting to MCP server: confluence at http://localhost:8083/mcp
   MCP server confluence: loaded 4 tools: ['confluence_search_pages', 'confluence_get_page', 'confluence_create_page', 'confluence_update_page']
   Successfully loaded 4 tools from server: confluence
   Total tools loaded: X
   Tool names: [..., 'confluence_search_pages', 'confluence_get_page', ...]
   ```

3. **Test the connection:**
   ```bash
   python test_confluence_mcp.py
   ```

### If Still Not Working

1. **Check environment variables:**
   ```bash
   # In your .env file, ensure:
   CONFLUENCE_MCP_SERVER_URL=http://localhost:8083/mcp
   ```

2. **Verify the server is running:**
   ```bash
   curl http://localhost:8083/mcp
   ```

3. **Check for port conflicts:**
   ```bash
   lsof -i :8083
   ```

4. **Review the logs:**
   - LangGraph server logs should show which tools were loaded
   - Confluence MCP server logs should show it started successfully
   - Look for any connection errors

### Enhanced Logging

The code now includes enhanced logging that will show:
- Which MCP servers are being connected to
- How many tools are loaded from each server
- The names of all loaded tools
- Any connection failures with full error details

Check the LangGraph server startup logs to see exactly what's happening.

