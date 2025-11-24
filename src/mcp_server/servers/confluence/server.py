"""FastMCP server for Confluence domain tools.

This module provides MCP tools for Confluence integration:
- confluence_search_pages: Search for Confluence pages
- confluence_get_page: Get page content
- confluence_create_page: Create a new page
- confluence_update_page: Update an existing page
"""

import logging

from fastmcp import FastMCP

from .schema.input import (
    ConfluenceCreatePageInput,
    ConfluenceGetPageInput,
    ConfluenceSearchPagesInput,
    ConfluenceUpdatePageInput,
)
from .usecases.create_page_usecase import CreatePageUseCase
from .usecases.get_page_usecase import GetPageUseCase
from .usecases.search_pages_usecase import SearchPagesUseCase
from .usecases.update_page_usecase import UpdatePageUseCase

logger = logging.getLogger(__name__)

# Create FastMCP instance for Confluence domain
confluence_mcp = FastMCP("Confluence Tools")

# Initialize use cases
_search_pages_usecase = SearchPagesUseCase()
_get_page_usecase = GetPageUseCase()
_create_page_usecase = CreatePageUseCase()
_update_page_usecase = UpdatePageUseCase()


@confluence_mcp.tool()
def confluence_search_pages(
    query: str,
    space_key: str | None = None,
    limit: int = 10,
) -> dict:
    """
    Search for Confluence pages.

    Args:
        query: Search query string
        space_key: Optional space key to limit search
        limit: Maximum number of results to return

    Returns:
        Dictionary with 'pages' list containing page information
    """
    payload = ConfluenceSearchPagesInput(query=query, space_key=space_key, limit=limit)
    result = _search_pages_usecase.execute(payload)
    return result.model_dump()


@confluence_mcp.tool()
def confluence_get_page(page_id: str) -> dict:
    """
    Get full content of a Confluence page.

    Args:
        page_id: The ID of the page to retrieve

    Returns:
        Dictionary with page content, title, and URL
    """
    payload = ConfluenceGetPageInput(page_id=page_id)
    result = _get_page_usecase.execute(payload)
    return result.model_dump()


@confluence_mcp.tool()
def confluence_create_page(
    space_key: str,
    title: str,
    body: str,
    parent_id: str | None = None,
) -> dict:
    """
    Create a new Confluence page.

    Args:
        space_key: The space key where the page should be created. Use "ANALYTICS" for the Analytics space.
        title: Page title
        body: Page body content (markdown, HTML, or Confluence Storage Format)
        parent_id: Optional parent page ID

    Returns:
        Dictionary with created page ID and URL
    """
    payload = ConfluenceCreatePageInput(
        space_key=space_key, title=title, body=body, parent_id=parent_id
    )
    result = _create_page_usecase.execute(payload)
    return result.model_dump()


@confluence_mcp.tool()
def confluence_update_page(
    page_id: str,
    title: str | None = None,
    body: str | None = None,
) -> dict:
    """
    Update an existing Confluence page.

    Args:
        page_id: The ID of the page to update
        title: Optional new title
        body: Optional new body content

    Returns:
        Dictionary with updated page information
    """
    payload = ConfluenceUpdatePageInput(page_id=page_id, title=title, body=body)
    result = _update_page_usecase.execute(payload)
    return result.model_dump()
