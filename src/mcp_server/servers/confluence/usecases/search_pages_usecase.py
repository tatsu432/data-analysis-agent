"""Use case for searching Confluence pages."""

import html
import logging
import os
import re

from ..infrastructure.confluence_client import ConfluenceClient
from ..schema.input import ConfluenceSearchPagesInput
from ..schema.output import ConfluenceSearchPagesOutput

logger = logging.getLogger(__name__)


class SearchPagesUseCase:
    """Use case for searching Confluence pages."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.confluence_client = ConfluenceClient()

    def execute(
        self, payload: ConfluenceSearchPagesInput
    ) -> ConfluenceSearchPagesOutput:
        """
        Search for Confluence pages.

        Args:
            payload: Input containing query, space_key, and limit

        Returns:
            ConfluenceSearchPagesOutput with pages list
        """
        try:
            client = self.confluence_client.get_client()

            # Build CQL (Confluence Query Language) query
            # Only search for pages (not attachments, comments, etc.)
            # Escape single quotes in query to prevent CQL injection
            escaped_query = payload.query.replace("'", "''")

            # For meta-questions with "*" or very broad queries, search all pages
            # For specific queries, search in title first (more restrictive, matches Content view better)
            if payload.query.strip() in ("*", ""):
                # Very broad query - just get all pages
                cql = "type = page"
            else:
                # Search in title first - this matches what users see in Content view
                cql = f"type = page AND title ~ '{escaped_query}'"

            if payload.space_key:
                cql += f" AND space = {payload.space_key}"

            # Search pages - request more results to account for filtering
            search_limit = payload.limit * 2  # Get more results to filter
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

                # Exclude archived pages
                status = content.get("status", "").lower()
                if status == "archived" or status == "trashed":
                    continue

                # Skip attachment-like entries
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
                if len(pages) >= payload.limit:
                    break

            logger.info(
                f"Found {len(pages)} valid Confluence pages (filtered from {len(results.get('results', []))} results)"
            )
            return ConfluenceSearchPagesOutput(
                pages=pages, results=pages, items=pages, error=None
            )
        except Exception as e:
            logger.error(f"Error searching Confluence: {e}", exc_info=True)
            return ConfluenceSearchPagesOutput(
                pages=[], results=[], items=[], error=str(e)
            )
