"""Use case for updating a Confluence page."""

import logging

from ..infrastructure.confluence_client import ConfluenceClient
from ..schema.input import ConfluenceUpdatePageInput
from ..schema.output import ConfluenceUpdatePageOutput

logger = logging.getLogger(__name__)


class UpdatePageUseCase:
    """Use case for updating a Confluence page."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.confluence_client = ConfluenceClient()

    def execute(self, payload: ConfluenceUpdatePageInput) -> ConfluenceUpdatePageOutput:
        """
        Update an existing Confluence page.

        Args:
            payload: Input containing page_id, optional title and body

        Returns:
            ConfluenceUpdatePageOutput with updated page information
        """
        try:
            client = self.confluence_client.get_client()

            # Get current page to preserve version
            current_page = client.get_page_by_id(payload.page_id, expand="version")
            version = current_page.get("version", {}).get("number", 1)

            # Update page
            result = client.update_page(
                page_id=payload.page_id,
                title=payload.title or current_page.get("title"),
                body=payload.body
                or current_page.get("body", {}).get("storage", {}).get("value"),
                version=version + 1,
                representation="storage",
            )

            page_url = result.get("_links", {}).get("webui", "")

            return ConfluenceUpdatePageOutput(
                page_id=payload.page_id,
                id=payload.page_id,  # Alias
                url=page_url,
                page_url=page_url,  # Alias
                pageUrl=page_url,  # Alias
                title=result.get("title"),
                error=None,
            )
        except Exception as e:
            logger.error(f"Error updating Confluence page: {e}")
            return ConfluenceUpdatePageOutput(
                page_id=payload.page_id,
                id=payload.page_id,
                url="",
                page_url="",
                pageUrl="",
                title="",
                error=str(e),
            )
