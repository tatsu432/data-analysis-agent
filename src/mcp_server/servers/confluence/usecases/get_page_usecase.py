"""Use case for getting a Confluence page."""

import logging

from ..infrastructure.confluence_client import ConfluenceClient
from ..schema.input import ConfluenceGetPageInput
from ..schema.output import ConfluenceGetPageOutput

logger = logging.getLogger(__name__)


class GetPageUseCase:
    """Use case for getting a Confluence page."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.confluence_client = ConfluenceClient()

    def execute(self, payload: ConfluenceGetPageInput) -> ConfluenceGetPageOutput:
        """
        Get full content of a Confluence page.

        Args:
            payload: Input containing page_id

        Returns:
            ConfluenceGetPageOutput with page content, title, and URL
        """
        try:
            client = self.confluence_client.get_client()

            # Get page content
            page = client.get_page_by_id(payload.page_id, expand="body.storage,version")

            # Extract content
            body = page.get("body", {}).get("storage", {}).get("value", "")
            title = page.get("title", "")
            url = page.get("_links", {}).get("webui", "")

            return ConfluenceGetPageOutput(
                page_id=payload.page_id,
                id=payload.page_id,  # Alias
                title=title,
                content=body,
                body=body,  # Alias
                text=body,  # Alias
                url=url,
                page_url=url,  # Alias
                pageUrl=url,  # Alias
                error=None,
            )
        except Exception as e:
            logger.error(f"Error getting Confluence page: {e}")
            return ConfluenceGetPageOutput(
                page_id=payload.page_id,
                id=payload.page_id,
                title="",
                content="",
                body="",
                text="",
                url="",
                page_url="",
                pageUrl="",
                error=str(e),
            )
