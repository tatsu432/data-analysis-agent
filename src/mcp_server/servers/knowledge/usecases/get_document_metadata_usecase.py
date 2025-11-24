"""Use case for getting document metadata."""

import logging

from ..infrastructure.document_store import DocumentStore
from ..infrastructure.knowledge_registry import DOCUMENTS
from ..schema.input import GetDocumentMetadataInput
from ..schema.output import DocumentMetadata

logger = logging.getLogger(__name__)


class GetDocumentMetadataUseCase:
    """Use case for getting document metadata."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.document_store = DocumentStore(DOCUMENTS)

    def execute(self, payload: GetDocumentMetadataInput) -> DocumentMetadata:
        """
        Returns metadata for the specified document.

        Args:
            payload: Input containing doc_id

        Returns:
            DocumentMetadata object

        Raises:
            ValueError: If doc_id is not found
        """
        logger.info("=" * 60)
        logger.info("USECASE EXECUTION: get_document_metadata")
        logger.info("=" * 60)
        logger.info(f"INPUT - doc_id: {payload.doc_id}")

        try:
            meta = self.document_store.validate_doc_id(payload.doc_id)
            result = DocumentMetadata(
                doc_id=payload.doc_id,
                title=meta.get("title", ""),
                kind=meta.get("kind", "other"),
                source_path=meta.get("source_path", ""),
                description=meta.get("description"),
                tags=meta.get("tags", []),
            )

            logger.info("OUTPUT SUMMARY:")
            logger.info(f"  - Title: {result.title}")
            logger.info(f"  - Kind: {result.kind}")
            logger.info("=" * 60)

            return result
        except ValueError as e:
            logger.error(str(e))
            raise

