# Confluence MCP Server

A FastMCP-based Confluence MCP server for the Data Analysis Agent.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** in your main project's `.env` file:
   ```bash
   CONFLUENCE_URL=https://yourcompany.atlassian.net
   CONFLUENCE_USERNAME=your.email@company.com
   CONFLUENCE_API_TOKEN=your_api_token_here
   CONFLUENCE_MCP_PORT=8083
   ```

3. **Run the server:**
   ```bash
   python server.py
   ```

The server will start on `http://localhost:8083` and expose tools at `http://localhost:8083/mcp`.

## Getting a Confluence API Token

1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Give it a label (e.g., "MCP Server")
4. Copy the token and add it to your `.env` file as `CONFLUENCE_API_TOKEN`

## Tools Provided

- `confluence_search_pages`: Search for Confluence pages
- `confluence_get_page`: Get full content of a page
- `confluence_create_page`: Create a new page
- `confluence_update_page`: Update an existing page

For more details, see the main project's `CONFLUENCE_MCP_SETUP.md` file.

