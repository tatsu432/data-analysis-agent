"""Knowledge index manager for lazy initialization."""

import logging

from .document_store import DocumentStore
from .knowledge_index import build_global_knowledge_index
from .knowledge_registry import DOCUMENTS

logger = logging.getLogger(__name__)

# Global knowledge index (lazy initialization)
_global_index = None


def ensure_index_built():
    """Ensure the global knowledge index is built."""
    global _global_index
    if _global_index is None:
        logger.info("Building knowledge index with hybrid search...")
        document_store = DocumentStore(DOCUMENTS)
        _global_index = build_global_knowledge_index(
            DOCUMENTS,
            document_store,
            use_embeddings=True,
            embedding_model=None,  # Auto-detect based on language
            use_hybrid_search=True,  # Use hybrid search for better results
        )
    return _global_index
