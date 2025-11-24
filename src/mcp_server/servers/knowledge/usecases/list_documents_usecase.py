"""Use case for listing available documents."""

import logging
from typing import Any, Dict, List

from ..infrastructure.knowledge_registry import DOCUMENTS

logger = logging.getLogger(__name__)


class ListDocumentsUseCase:
    """Use case for listing all available documents in the knowledge base."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.documents = DOCUMENTS

    def execute(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns a list of available documents with their metadata.

        Returns:
            Dictionary with a "documents" key containing list of document metadata
        """
        logger.info("=" * 60)
        logger.info("USECASE EXECUTION: list_documents")
        logger.info("=" * 60)

        documents_list = []
        for doc_id, meta in self.documents.items():
            doc_info = {
                "doc_id": doc_id,
                "title": meta.get("title", ""),
                "kind": meta.get("kind", "other"),
                "source_path": meta.get("source_path", ""),
                "description": meta.get("description"),
                "tags": meta.get("tags", []),
            }
            documents_list.append(doc_info)

        result = {"documents": documents_list}

        logger.info(f"OUTPUT SUMMARY: {len(documents_list)} documents available")
        logger.debug(f"  - Document IDs: {[doc['doc_id'] for doc in documents_list]}")
        logger.info("=" * 60)

        return result

