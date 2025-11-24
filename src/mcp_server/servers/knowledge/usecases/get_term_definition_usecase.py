"""Use case for getting term definitions from knowledge base."""

import logging
from typing import Any, Dict, Optional

from ..infrastructure.knowledge_index_manager import ensure_index_built
from ..schema.input import GetTermDefinitionInput

logger = logging.getLogger(__name__)


class GetTermDefinitionUseCase:
    """Use case for getting term definitions from the knowledge base."""

    def __init__(self) -> None:
        """Initialize the use case."""
        pass

    def execute(self, payload: GetTermDefinitionInput) -> Optional[Dict[str, Any]]:
        """
        Get the definition of a term from the knowledge base.

        First tries exact match (case-insensitive), then searches synonyms,
        then falls back to similarity search.

        Args:
            payload: Input containing term

        Returns:
            TermEntry as dictionary if found, None otherwise
        """
        logger.info("=" * 60)
        logger.info("USECASE EXECUTION: get_term_definition")
        logger.info("=" * 60)
        logger.info(f"INPUT - term: {payload.term}")

        index = ensure_index_built()

        # Normalize term
        term_normalized = payload.term.strip().lower()

        # First, try exact match in term entries
        for entry in index.term_entries:
            if entry.term.lower() == term_normalized:
                logger.info(f"Found exact match: {entry.term}")
                logger.info("=" * 60)
                return entry.model_dump()

            # Check synonyms
            for synonym in entry.synonyms:
                if synonym.lower() == term_normalized:
                    logger.info(f"Found match in synonyms: {entry.term}")
                    logger.info("=" * 60)
                    return entry.model_dump()

        # Fallback to similarity search
        logger.info("No exact match found, trying similarity search...")
        hits = index.search(query=payload.term, scopes=["terms"], top_k=1)

        if hits and hits[0].term_entry:
            entry = hits[0].term_entry
            logger.info(
                f"Found similar match: {entry.term} (score: {hits[0].score:.3f})"
            )
            logger.info("=" * 60)
            return entry.model_dump()

        logger.info("No match found")
        logger.info("=" * 60)
        return None
