"""Use case for creating a Confluence page."""

import logging

from ..infrastructure.confluence_client import ConfluenceClient
from ..schema.input import ConfluenceCreatePageInput
from ..schema.output import ConfluenceCreatePageOutput

logger = logging.getLogger(__name__)


class CreatePageUseCase:
    """Use case for creating a Confluence page."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.confluence_client = ConfluenceClient()

    def execute(self, payload: ConfluenceCreatePageInput) -> ConfluenceCreatePageOutput:
        """
        Create a new Confluence page.

        Args:
            payload: Input containing space_key, title, body, and optional parent_id

        Returns:
            ConfluenceCreatePageOutput with created page ID and URL
        """
        try:
            client = self.confluence_client.get_client()

            # Try to convert markdown to HTML if it looks like markdown
            # Simple heuristic: if body contains markdown syntax, convert it
            body_to_use = payload.body
            if (
                "```" in payload.body
                or payload.body.strip().startswith("#")
                or "**" in payload.body
            ):
                try:
                    import markdown

                    # Convert markdown to HTML
                    body_to_use = markdown.markdown(
                        payload.body, extensions=["fenced_code", "tables", "codehilite"]
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
                    space=payload.space_key,
                    title=payload.title,
                    body=body_to_use,
                    parent_id=payload.parent_id,
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
                        space=payload.space_key,
                        title=payload.title,
                        body=payload.body,  # Use original body for wiki format
                        parent_id=payload.parent_id,
                        type="page",
                        representation="wiki",  # Wiki markup format
                    )
                except Exception as wiki_error:
                    logger.error(
                        f"Failed to create page with wiki format: {wiki_error}"
                    )
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

            return ConfluenceCreatePageOutput(
                page_id=page_id,
                id=page_id,  # Alias
                url=page_url,
                page_url=page_url,  # Alias
                pageUrl=page_url,  # Alias
                title=payload.title,
                error=None,
            )
        except Exception as e:
            logger.error(f"Error creating Confluence page: {e}", exc_info=True)
            return ConfluenceCreatePageOutput(
                page_id="",
                id="",
                url="",
                page_url="",
                pageUrl="",
                title=payload.title,
                error=str(e),
            )
