"""MCP tools for Confluence integration."""

import logging
import os
from typing import Any, Dict, Optional

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Create FastMCP instance for Confluence tools
confluence_mcp = FastMCP("Confluence Tools")

# Global Confluence client (will be initialized on first use)
_confluence_client = None


def get_confluence_client():
    """Get or create Confluence client."""
    global _confluence_client
    if _confluence_client is None:
        try:
            from atlassian import Confluence
        except ImportError:
            raise ImportError(
                "atlassian-python-api is required. Install with: pip install atlassian-python-api"
            )

        url = os.getenv("CONFLUENCE_URL", "")
        username = os.getenv("CONFLUENCE_USERNAME", "")
        api_token = os.getenv("CONFLUENCE_API_TOKEN", "")

        if not all([url, username, api_token]):
            raise ValueError(
                "Missing Confluence credentials. Set CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN"
            )

        _confluence_client = Confluence(
            url=url,
            username=username,
            password=api_token,  # API token is used as password
            cloud=True,  # Set to False for Confluence Server/Data Center
        )
        logger.info(f"Initialized Confluence client for {url}")

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
        import html
        import re

        client = get_confluence_client()

        # Build CQL (Confluence Query Language) query
        # Only search for pages (not attachments, comments, etc.)
        # Escape single quotes in query to prevent CQL injection
        escaped_query = query.replace("'", "''")

        # For meta-questions with "*" or very broad queries, search all pages
        # For specific queries, search in title first (more restrictive, matches Content view better)
        # The Content view typically shows pages by title, not by body content matches
        if query.strip() in ("*", ""):
            # Very broad query - just get all pages
            cql = "type = page"
        else:
            # Search in title first - this matches what users see in Content view
            # If you want to also search body, use: title ~ '{escaped_query}' OR text ~ '{escaped_query}'
            # But for now, prioritize title to match Content view behavior
            cql = f"type = page AND title ~ '{escaped_query}'"

        if space_key:
            cql += f" AND space = {space_key}"

        # Search pages - request more results to account for filtering
        search_limit = limit * 2  # Get more results to filter
        results = client.cql(cql, limit=search_limit)

        pages = []
        seen_ids = set()  # Track seen page IDs to avoid duplicates

        for result in results.get("results", []):
            content = result.get("content", {})
            content_id = content.get("id")
            content_type = content.get("type", "").lower()
            title = content.get("title", "").strip()

            # Skip invalid entries
            if not content_id or not title:
                continue

            # Skip if we've already seen this page
            if content_id in seen_ids:
                continue

            # Only include actual pages (not attachments, comments, etc.)
            if content_type != "page":
                continue

            # Exclude archived pages (they have status = "archived")
            # This matches what the Content view shows
            status = content.get("status", "").lower()
            if status == "archived" or status == "trashed":
                continue

            # Skip attachment-like entries (IDs starting with "att" or titles that look like filenames)
            if str(content_id).startswith("att") or re.match(
                r"^[\w\-]+\.(jpg|jpeg|png|gif|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|tar|gz)(\?.*)?$",
                title,
                re.IGNORECASE,
            ):
                continue

            seen_ids.add(content_id)

            # Get URL
            url = content.get("_links", {}).get("webui", "")
            # Make URL absolute if it's relative
            if url and not url.startswith("http"):
                base_url = os.getenv("CONFLUENCE_URL", "").rstrip("/")
                url = f"{base_url}{url}" if base_url else url

            # Get space key
            space_info = content.get("space", {})
            space_key_result = (
                space_info.get("key") if isinstance(space_info, dict) else None
            )

            # Clean excerpt - decode HTML entities and strip
            excerpt = result.get("excerpt", "")
            if excerpt:
                # Decode HTML entities
                excerpt = html.unescape(excerpt)
                # Remove HTML tags
                excerpt = re.sub(r"<[^>]+>", "", excerpt)
                # Clean up whitespace
                excerpt = " ".join(excerpt.split())
                # Truncate if too long
                if len(excerpt) > 300:
                    excerpt = excerpt[:297] + "..."

            page_info = {
                "id": content_id,
                "page_id": content_id,  # Alias for compatibility
                "title": title,
                "name": title,  # Alias for compatibility
                "url": url,
                "page_url": url,  # Alias
                "pageUrl": url,  # Alias
                "space_key": space_key_result,
                "excerpt": excerpt,
                "snippet": excerpt,  # Alias
                "body": excerpt,  # Alias for search results
            }
            pages.append(page_info)

            # Stop if we have enough valid pages
            if len(pages) >= limit:
                break

        logger.info(
            f"Found {len(pages)} valid Confluence pages (filtered from {len(results.get('results', []))} results)"
        )
        return {"pages": pages, "results": pages, "items": pages}  # Multiple aliases
    except Exception as e:
        logger.error(f"Error searching Confluence: {e}", exc_info=True)
        return {"pages": [], "results": [], "items": [], "error": str(e)}


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
            "id": page_id,  # Alias
            "title": title,
            "content": body,
            "body": body,  # Alias
            "text": body,  # Alias
            "url": url,
            "page_url": url,  # Alias
            "pageUrl": url,  # Alias
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
        space_key: The space key where the page should be created. Use "ANALYTICS" for the Analytics space.
        title: Page title
        body: Page body content (markdown, HTML, or Confluence Storage Format)
        parent_id: Optional parent page ID

    Returns:
        Dictionary with created page ID and URL
    """
    try:
        client = get_confluence_client()

        # Try to convert markdown to HTML if it looks like markdown
        # Simple heuristic: if body contains markdown syntax, convert it
        body_to_use = body
        if "```" in body or body.strip().startswith("#") or "**" in body:
            try:
                import markdown

                # Convert markdown to HTML
                body_to_use = markdown.markdown(
                    body, extensions=["fenced_code", "tables", "codehilite"]
                )
                logger.info("Converted markdown to HTML for Confluence page")
            except ImportError:
                logger.warning("markdown library not available, using body as-is")
            except Exception as e:
                logger.warning(
                    f"Failed to convert markdown to HTML: {e}, using body as-is"
                )

        # Try creating with storage format first (HTML)
        # If that fails, try wiki format
        try:
            result = client.create_page(
                space=space_key,
                title=title,
                body=body_to_use,
                parent_id=parent_id,
                type="page",
                representation="storage",  # HTML/storage format
            )
        except Exception as storage_error:
            logger.warning(
                f"Failed to create page with storage format: {storage_error}"
            )
            logger.info("Trying with wiki format instead...")
            # Try with wiki format as fallback
            try:
                result = client.create_page(
                    space=space_key,
                    title=title,
                    body=body,  # Use original body for wiki format
                    parent_id=parent_id,
                    type="page",
                    representation="wiki",  # Wiki markup format
                )
            except Exception as wiki_error:
                logger.error(f"Failed to create page with wiki format: {wiki_error}")
                raise storage_error  # Raise the original error

        page_id = result.get("id")
        page_url = result.get("_links", {}).get("webui", "")

        if not page_id:
            logger.error(
                f"Page creation succeeded but no ID returned. Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}"
            )
            logger.error(f"Full result: {result}")

        logger.info(
            f"Successfully created Confluence page: id={page_id}, url={page_url}"
        )

        return {
            "page_id": page_id,
            "id": page_id,  # Alias
            "url": page_url,
            "page_url": page_url,  # Alias
            "pageUrl": page_url,  # Alias
            "title": title,
        }
    except Exception as e:
        logger.error(f"Error creating Confluence page: {e}", exc_info=True)
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

        page_url = result.get("_links", {}).get("webui", "")

        return {
            "page_id": page_id,
            "id": page_id,  # Alias
            "url": page_url,
            "page_url": page_url,  # Alias
            "pageUrl": page_url,  # Alias
            "title": result.get("title"),
        }
    except Exception as e:
        logger.error(f"Error updating Confluence page: {e}")
        return {"error": str(e)}
