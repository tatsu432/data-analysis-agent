"""MCP tools for knowledge base access (document dictionaries and manuals)."""

import logging
from typing import Any, Dict, List, Optional

from .analysis_tools import analysis_mcp
from .document_store import DocumentStore
from .knowledge_index import (
    build_global_knowledge_index,
)
from .knowledge_registry import DOCUMENTS
from .schema import DocumentMetadata

logger = logging.getLogger(__name__)

# Initialize document store
document_store = DocumentStore(DOCUMENTS)

# Build global knowledge index on module load
_global_index = None


def _ensure_index_built():
    """Ensure the global knowledge index is built."""
    global _global_index
    if _global_index is None:
        logger.info("Building knowledge index with hybrid search...")
        _global_index = build_global_knowledge_index(
            DOCUMENTS,
            document_store,
            use_embeddings=True,
            embedding_model=None,  # Auto-detect based on language
            use_hybrid_search=True,  # Use hybrid search for better results
        )
    return _global_index


@analysis_mcp.tool(
    name="list_documents",
    description="List all available documents in the knowledge base (Excel dictionaries and PDF manuals).",
)
def list_documents() -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a list of available documents with their metadata.

    Returns:
        Dictionary with a "documents" key containing list of document metadata
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: list_documents")
    logger.info("=" * 60)

    documents_list = []
    for doc_id, meta in DOCUMENTS.items():
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


@analysis_mcp.tool(
    name="get_document_metadata",
    description="Get metadata for a specific document by its ID.",
)
def get_document_metadata(doc_id: str) -> DocumentMetadata:
    """
    Returns metadata for the specified document.

    Args:
        doc_id: ID of the document to get metadata for

    Returns:
        DocumentMetadata object

    Raises:
        ValueError: If doc_id is not found
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: get_document_metadata")
    logger.info("=" * 60)
    logger.info(f"INPUT - doc_id: {doc_id}")

    try:
        meta = document_store.validate_doc_id(doc_id)
        result = DocumentMetadata(
            doc_id=doc_id,
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


@analysis_mcp.tool(
    name="get_term_definition",
    description=(
        "Get the definition of a specific term from the knowledge base. "
        "Searches for exact matches first, then falls back to similarity search. "
        "Returns the best matching term entry if found."
    ),
)
def get_term_definition(term: str) -> Optional[Dict[str, any]]:
    """
    Get the definition of a term from the knowledge base.

    First tries exact match (case-insensitive), then searches synonyms,
    then falls back to similarity search.

    Args:
        term: The term to look up

    Returns:
        TermEntry as dictionary if found, None otherwise
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: get_term_definition")
    logger.info("=" * 60)
    logger.info(f"INPUT - term: {term}")

    index = _ensure_index_built()

    # Normalize term
    term_normalized = term.strip().lower()

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
    hits = index.search(query=term, scopes=["terms"], top_k=1)

    if hits and hits[0].term_entry:
        entry = hits[0].term_entry
        logger.info(f"Found similar match: {entry.term} (score: {hits[0].score:.3f})")
        logger.info("=" * 60)
        return entry.model_dump()

    logger.info("No match found")
    logger.info("=" * 60)
    return None


@analysis_mcp.tool(
    name="search_knowledge",
    description=(
        "Search the knowledge base for terms and document chunks. "
        "Returns relevant term definitions and document excerpts ranked by similarity. "
        "Use this to find information about domain-specific terms, concepts, or topics."
    ),
)
def search_knowledge(
    query: str,
    scopes: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, List[Dict[str, any]]]:
    """
    Search the knowledge base for relevant information.

    Args:
        query: Search query string
        scopes: List of scopes to search - ["terms"] for term definitions only,
                ["docs"] for document chunks only, or ["terms", "docs"] for both.
                Defaults to both.
        top_k: Maximum number of results to return per scope (default: 5)

    Returns:
        Dictionary with "hits" key containing list of KnowledgeHit objects
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: search_knowledge")
    logger.info("=" * 60)
    logger.info(f"INPUT - query: {query}")
    logger.info(f"INPUT - scopes: {scopes}")
    logger.info(f"INPUT - top_k: {top_k}")

    if scopes is None:
        scopes = ["terms", "docs"]

    # Validate scopes
    valid_scopes = ["terms", "docs"]
    invalid_scopes = [s for s in scopes if s not in valid_scopes]
    if invalid_scopes:
        raise ValueError(
            f"Invalid scopes: {invalid_scopes}. Valid scopes are: {valid_scopes}"
        )

    index = _ensure_index_built()
    hits = index.search(query=query, scopes=scopes, top_k=top_k)

    # Convert to dictionaries
    hits_dict = [hit.model_dump() for hit in hits]

    result = {"hits": hits_dict}

    logger.info(f"OUTPUT SUMMARY: {len(hits)} results found")
    logger.debug(f"  - Results: {[(h.kind, h.score) for h in hits]}")
    logger.info("=" * 60)

    return result
