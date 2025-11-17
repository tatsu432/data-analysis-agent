# Confluence MCP Server Setup Guide

This guide provides step-by-step instructions for setting up a Confluence MCP server to work with the Data Analysis Agent.

## Overview

You have three main options for setting up a Confluence MCP server:

1. **Option 1: Create a Simple FastMCP Server** (Recommended for this project)
   - Most consistent with your existing architecture
   - Full control over implementation
   - Requires Confluence API credentials

2. **Option 2: Use Atlassian's Official Remote MCP Server**
   - Official Atlassian solution
   - Requires OAuth setup
   - Uses remote proxy

3. **Option 3: Use CData's MCP Server**
   - Commercial solution
   - Easy setup
   - Requires license

---

## Option 1: Create a Simple FastMCP Confluence Server (Recommended)

This option creates a Confluence MCP server similar to your existing data analysis MCP server.

### Prerequisites

1. **Confluence API Access**
   - A Confluence Cloud site (e.g., `https://yourcompany.atlassian.net`)
   - An API token: [Create one here](https://id.atlassian.com/manage-profile/security/api-tokens)
   - Or username/password for Confluence Server/Data Center

2. **Python Dependencies**
   ```bash
   pip install fastmcp atlassian-python-api
   ```

### Step 1: Create the Confluence MCP Server Directory

Create a new directory for the Confluence MCP server (separate from your data analysis server):

```bash
mkdir -p confluence_mcp_server
cd confluence_mcp_server
```

### Step 2: Create the Confluence Tools File

Create `confluence_tools.py`:

```python
"""MCP tools for Confluence integration."""

import logging
from typing import Any, Dict, List, Optional

from atlassian import Confluence
from fastmcp import FastMCP
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create FastMCP instance
confluence_mcp = FastMCP("Confluence MCP Server")


class ConfluenceSettings(BaseModel):
    """Settings for Confluence connection."""
    
    url: str  # e.g., "https://yourcompany.atlassian.net"
    username: str  # Your email or username
    api_token: str  # API token from Atlassian
    # OR for Confluence Server:
    # password: str  # Instead of api_token


# Global Confluence client (will be initialized on startup)
_confluence_client: Optional[Confluence] = None


def get_confluence_client() -> Confluence:
    """Get or create Confluence client."""
    global _confluence_client
    if _confluence_client is None:
        import os
        settings = ConfluenceSettings(
            url=os.getenv("CONFLUENCE_URL", ""),
            username=os.getenv("CONFLUENCE_USERNAME", ""),
            api_token=os.getenv("CONFLUENCE_API_TOKEN", ""),
        )
        _confluence_client = Confluence(
            url=settings.url,
            username=settings.username,
            password=settings.api_token,  # API token is used as password
            cloud=True,  # Set to False for Confluence Server/Data Center
        )
    return _confluence_client


@confluence_mcp.tool()
def confluence_search_pages(
    query: str,
    space_key: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search for Confluence pages.
    
    Args:
        query: Search query string
        space_key: Optional space key to limit search
        limit: Maximum number of results to return
    
    Returns:
        Dictionary with 'pages' list containing page information
    """
    try:
        client = get_confluence_client()
        
        # Build CQL (Confluence Query Language) query
        cql = f"text ~ '{query}'"
        if space_key:
            cql += f" AND space = {space_key}"
        
        # Search pages
        results = client.cql(cql, limit=limit)
        
        pages = []
        for result in results.get("results", []):
            page_info = {
                "id": result.get("content", {}).get("id"),
                "title": result.get("content", {}).get("title"),
                "url": result.get("content", {}).get("_links", {}).get("webui"),
                "space_key": result.get("content", {}).get("space", {}).get("key"),
                "excerpt": result.get("excerpt", ""),
            }
            pages.append(page_info)
        
        return {"pages": pages}
    except Exception as e:
        logger.error(f"Error searching Confluence: {e}")
        return {"pages": [], "error": str(e)}


@confluence_mcp.tool()
def confluence_get_page(page_id: str) -> Dict[str, Any]:
    """
    Get full content of a Confluence page.
    
    Args:
        page_id: The ID of the page to retrieve
    
    Returns:
        Dictionary with page content, title, and URL
    """
    try:
        client = get_confluence_client()
        
        # Get page content
        page = client.get_page_by_id(page_id, expand="body.storage,version")
        
        # Extract content
        body = page.get("body", {}).get("storage", {}).get("value", "")
        title = page.get("title", "")
        url = page.get("_links", {}).get("webui", "")
        
        return {
            "page_id": page_id,
            "title": title,
            "content": body,
            "url": url,
        }
    except Exception as e:
        logger.error(f"Error getting Confluence page: {e}")
        return {"error": str(e)}


@confluence_mcp.tool()
def confluence_create_page(
    space_key: str,
    title: str,
    body: str,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new Confluence page.
    
    Args:
        space_key: The space key where the page should be created
        title: Page title
        body: Page body content (in Confluence Storage Format or HTML)
        parent_id: Optional parent page ID
    
    Returns:
        Dictionary with created page ID and URL
    """
    try:
        client = get_confluence_client()
        
        # Create page
        result = client.create_page(
            space=space_key,
            title=title,
            body=body,
            parent_id=parent_id,
            type="page",
            representation="storage",  # or "wiki" or "storage"
        )
        
        page_id = result.get("id")
        page_url = result.get("_links", {}).get("webui", "")
        
        return {
            "page_id": page_id,
            "url": page_url,
            "title": title,
        }
    except Exception as e:
        logger.error(f"Error creating Confluence page: {e}")
        return {"error": str(e)}


@confluence_mcp.tool()
def confluence_update_page(
    page_id: str,
    title: Optional[str] = None,
    body: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing Confluence page.
    
    Args:
        page_id: The ID of the page to update
        title: Optional new title
        body: Optional new body content
    
    Returns:
        Dictionary with updated page information
    """
    try:
        client = get_confluence_client()
        
        # Get current page to preserve version
        current_page = client.get_page_by_id(page_id, expand="version")
        version = current_page.get("version", {}).get("number", 1)
        
        # Update page
        result = client.update_page(
            page_id=page_id,
            title=title or current_page.get("title"),
            body=body or current_page.get("body", {}).get("storage", {}).get("value"),
            version=version + 1,
            representation="storage",
        )
        
        return {
            "page_id": page_id,
            "url": result.get("_links", {}).get("webui", ""),
            "title": result.get("title"),
        }
    except Exception as e:
        logger.error(f"Error updating Confluence page: {e}")
        return {"error": str(e)}
```

### Step 3: Create the Server Entry Point

Create `server.py`:

```python
"""Confluence MCP server entry point."""

import asyncio
import logging
import os
from dotenv import load_dotenv

from confluence_tools import confluence_mcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def main() -> None:
    """Main function."""
    try:
        # Validate required environment variables
        required_vars = ["CONFLUENCE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        logger.info("Starting Confluence MCP server...")
        logger.info(f"Confluence URL: {os.getenv('CONFLUENCE_URL')}")
        
        # Run the MCP server
        await confluence_mcp.run_async(
            transport="http",  # or "streamable-http" for SSE
            host="0.0.0.0",
            port=int(os.getenv("CONFLUENCE_MCP_PORT", "8083")),
        )
    except Exception as e:
        logger.error(f"Failed to run Confluence MCP server: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 4: Create Requirements File

Create `requirements.txt`:

```txt
fastmcp
atlassian-python-api
python-dotenv
```

### Step 5: Set Up Environment Variables

Add to your main project's `.env` file (or create a separate one for the Confluence server):

```bash
# Confluence MCP Server Configuration
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_USERNAME=your.email@company.com
CONFLUENCE_API_TOKEN=your_api_token_here
CONFLUENCE_MCP_PORT=8083

# For your main project (already exists)
CONFLUENCE_MCP_SERVER_URL=http://localhost:8083/mcp
```

### Step 6: Install Dependencies and Run

```bash
# In the confluence_mcp_server directory
pip install -r requirements.txt

# Run the server
python server.py
```

The server will start on `http://localhost:8083` and expose tools at `http://localhost:8083/mcp`.

### Step 7: Test the Server

You can test the server is working by checking the tools endpoint:

```bash
curl http://localhost:8083/mcp
```

---

## Option 2: Use Atlassian's Official Remote MCP Server

This uses Atlassian's hosted MCP server with OAuth authentication.

### Prerequisites

- Node.js 18+ installed
- Atlassian Cloud account

### Steps

1. **Install the MCP Remote Proxy:**
   ```bash
   npm install -g @modelcontextprotocol/server-remote
   ```

2. **Start the Proxy:**
   ```bash
   npx -y mcp-remote https://mcp.atlassian.com/v1/sse
   ```
   
   This will:
   - Open a browser for OAuth authentication
   - Start a local proxy server (usually on a random port)

3. **Configure Your Project:**
   
   Since Atlassian's server uses a remote proxy, you'll need to either:
   - Use the proxy URL directly (if it exposes HTTP endpoints)
   - Or create a local wrapper that connects to the proxy
   
   Update your `.env`:
   ```bash
   CONFLUENCE_MCP_SERVER_URL=http://localhost:<proxy-port>/mcp
   ```

**Note:** This option is more complex to integrate with HTTP-based MCP clients. Option 1 is recommended for your use case.

---

## Option 3: Use CData's MCP Server

CData provides a commercial MCP server for Confluence.

### Steps

1. **Download:** Visit [CData Confluence MCP](https://www.cdata.com/drivers/confluence/download/mcp/)
2. **Install:** Follow their installation instructions
3. **Configure:** Set up connection to your Confluence instance
4. **Start:** Run their server
5. **Configure:** Point your project to their server URL

---

## Recommended Setup for Your Project

**I recommend Option 1** because:

1. ✅ Consistent with your existing FastMCP architecture
2. ✅ Full control over tool implementation
3. ✅ Easy to customize and extend
4. ✅ Works seamlessly with your HTTP-based MCP client
5. ✅ No external dependencies or licensing

### Quick Start (Option 1)

1. Create the `confluence_mcp_server` directory in your project root
2. Copy the code files above
3. Install dependencies: `pip install fastmcp atlassian-python-api python-dotenv`
4. Add Confluence credentials to `.env`
5. Run: `python confluence_mcp_server/server.py`
6. Update your main `.env` with `CONFLUENCE_MCP_SERVER_URL=http://localhost:8083/mcp`

### Running Both Servers

You'll need to run both servers:

**Terminal 1 - Data Analysis MCP Server:**
```bash
python -m src.mcp_server
# Runs on port 8082
```

**Terminal 2 - Confluence MCP Server:**
```bash
python confluence_mcp_server/server.py
# Runs on port 8083
```

**Terminal 3 - LangGraph Server:**
```bash
langgraph dev --config src/langgraph_server/langgraph.json
```

**Terminal 4 - Streamlit UI:**
```bash
streamlit run src/app/ui.py
```

---

## Troubleshooting

### Issue: "Confluence MCP tools are not available"

**Solution:** 
- Ensure the Confluence MCP server is running
- Check `CONFLUENCE_MCP_SERVER_URL` is set correctly
- Verify the server is accessible: `curl http://localhost:8083/mcp`

### Issue: Authentication Errors

**Solution:**
- Verify your Confluence API token is correct
- Check that your username is your email address (for Cloud)
- Ensure you have appropriate permissions in Confluence

### Issue: Tool Not Found Errors

**Solution:**
- Check that tool names match what the agent expects
- The agent looks for tools containing: `confluence`, `search`, `get`, `create`, `page`
- Verify tools are registered in the FastMCP instance

---

## Next Steps

Once your Confluence MCP server is running:

1. Test it with a simple query: "What pages exist in Confluence?"
2. Try exporting an analysis: "Create a Confluence report from this analysis"
3. Try reading from Confluence: "Summarize the latest analysis report in Confluence"

For more details, see the main README.md section on "Confluence Integration via MCP".

