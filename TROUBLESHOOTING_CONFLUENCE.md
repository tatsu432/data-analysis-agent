# Troubleshooting Confluence Integration

## Issue: "Confluence tools not found"

### Root Cause
Confluence tools are part of the unified MCP server and are automatically enabled when Confluence credentials are configured. If tools are not found, it's likely due to missing credentials or the unified server not running.

### Fix Steps

1. **Check Confluence Credentials:**
   ```bash
   # Ensure these are set in your .env file:
   CONFLUENCE_URL=https://yourcompany.atlassian.net
   CONFLUENCE_USERNAME=your.email@company.com
   CONFLUENCE_API_TOKEN=your_api_token_here
   ```

2. **Restart the Unified MCP Server:**
   ```bash
   # Stop the current server (Ctrl+C) and restart:
   python -m src.mcp_server
   ```

3. **Check the logs when starting the unified MCP server:**
   You should see logs like:
   ```
   ✅ Successfully imported analysis tools
   ✅ Successfully imported Confluence tools
   📦 Successfully configured unified MCP server with 2 domain(s): analysis, confluence
   ```

4. **Check the logs when starting the LangGraph server:**
   You should see logs like:
   ```
   ✅ Unified MCP server configured: http://localhost:8082/mcp (includes: analysis, knowledge, confluence tools)
   ✅ Successfully loaded X tools from server: unified
   ✅ Confluence tools found: ['confluence_search_pages', 'confluence_get_page', ...]
   ```

5. **Test the connection:**
   ```bash
   python test_confluence_mcp.py
   ```

### If Still Not Working

1. **Check environment variables:**
   ```bash
   # In your .env file, ensure:
   MCP_SERVER_URL=http://localhost:8082/mcp
   CONFLUENCE_URL=https://yourcompany.atlassian.net
   CONFLUENCE_USERNAME=your.email@company.com
   CONFLUENCE_API_TOKEN=your_api_token_here
   ```

2. **Verify the unified server is running:**
   ```bash
   curl http://localhost:8082/mcp
   ```

3. **Check for port conflicts:**
   ```bash
   lsof -i :8082
   ```

4. **Run diagnostic script:**
   ```bash
   python check_confluence_config.py
   ```

## Issue: "Confluence credentials not configured"

### Solution
Confluence tools are optional and only enabled when credentials are provided. If you don't need Confluence integration, you can ignore this warning. The unified MCP server will still work with analysis and knowledge tools.

If you want to enable Confluence tools:
1. Get your Confluence API token from: https://id.atlassian.com/manage-profile/security/api-tokens
2. Add credentials to `.env` file
3. Restart the unified MCP server

## Architecture Notes

- **Unified Server**: All tools (analysis, knowledge, confluence) are exposed through a single MCP server
- **Automatic Detection**: Confluence tools are automatically included when credentials are configured
- **No Separate Server**: Unlike the old architecture, there's no separate Confluence MCP server to manage
