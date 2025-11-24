"""Use case for searching the knowledge base."""

import logging
from typing import Any, Dict, List

from ..infrastructure.knowledge_index_manager import ensure_index_built
from ..schema.input import SearchKnowledgeInput

logger = logging.getLogger(__name__)


class SearchKnowledgeUseCase:
    """Use case for searching the knowledge base."""

    def __init__(self) -> None:
        """Initialize the use case."""
        pass

    def execute(self, payload: SearchKnowledgeInput) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search the knowledge base for relevant information.

        Args:
            payload: Input containing query, scopes, and top_k

        Returns:
            Dictionary with "hits" key containing list of KnowledgeHit objects
        """
        logger.info("=" * 60)
        logger.info("USECASE EXECUTION: search_knowledge")
        logger.info("=" * 60)
        logger.info(f"INPUT - query: {payload.query}")
        logger.info(f"INPUT - scopes: {payload.scopes}")
        logger.info(f"INPUT - top_k: {payload.top_k}")

        scopes = payload.scopes if payload.scopes is not None else ["terms", "docs"]

        # Validate scopes
        valid_scopes = ["terms", "docs"]
        invalid_scopes = [s for s in scopes if s not in valid_scopes]
        if invalid_scopes:
            raise ValueError(
                f"Invalid scopes: {invalid_scopes}. Valid scopes are: {valid_scopes}"
            )

        index = ensure_index_built()
        hits = index.search(query=payload.query, scopes=scopes, top_k=payload.top_k)

        # Convert to dictionaries
        hits_dict = [hit.model_dump() for hit in hits]

        result = {"hits": hits_dict}

        logger.info(f"OUTPUT SUMMARY: {len(hits)} results found")
        logger.debug(f"  - Results: {[(h.kind, h.score) for h in hits]}")
        logger.info("=" * 60)

        return result
