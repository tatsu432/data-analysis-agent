"""MCP tools for knowledge base access (document dictionaries and manuals)."""

import logging
from typing import Any, Dict, List, Optional

from .analysis_tools import analysis_mcp
from .document_store import DocumentStore
from .knowledge_index import (
    build_global_knowledge_index,
)
from .knowledge_registry import DOCUMENTS
from .schema import (
    DocumentMetadata,
    GetDocumentMetadataInput,
    GetTermDefinitionInput,
    SearchKnowledgeInput,
)

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
    description=(
        "List all available documents in the knowledge base. "
        "Returns metadata for Excel dictionaries and PDF manuals including doc_id, title, kind, source_path, description, and tags. "
        "Use this to discover what knowledge documents are available before searching or looking up terms. "
        "The knowledge base contains domain-specific terminology, definitions, and documentation that can help understand dataset columns and domain concepts."
    ),
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
    description=(
        "Get detailed metadata for a specific document in the knowledge base. "
        "Returns document information including doc_id, title, kind (excel_dictionary, pdf_manual, or other), "
        "source_path, description, and tags. "
        "Use this to get more information about a document after discovering it via list_documents(). "
        "The metadata helps understand what type of information the document contains."
    ),
)
def get_document_metadata(payload: GetDocumentMetadataInput) -> DocumentMetadata:
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
    logger.info(f"INPUT - doc_id: {payload.doc_id}")

    try:
        meta = document_store.validate_doc_id(payload.doc_id)
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


@analysis_mcp.tool(
    name="get_term_definition",
    description=(
        "Get the definition of a specific domain term from the knowledge base. "
        "This tool is essential for understanding domain-specific terminology that appears in datasets. "
        "Searches for exact matches first (case-insensitive), then checks synonyms, "
        "then falls back to similarity search if no exact match is found. "
        "Returns the best matching term entry with definition, synonyms, related_columns (dataset columns that relate to this term), "
        "source_doc_id, page number, and extra_context. "
        "\n\n"
        "Use this tool when you encounter domain-specific terms like: "
        "- 'GP' (General Practitioner), 'HP' (Hospital), 'TRx' (Total Prescriptions), 'Rx' (Prescriptions), "
        "- 'at-risk', 'DDI' (Drug-Drug Interaction), 'MR activity', 'unmet medical needs', etc. "
        "\n\n"
        "The related_columns field is particularly useful as it tells you which dataset columns map to this term, "
        "helping you write correct filtering and analysis code. "
        "Example: If 'GP' has related_columns=['channel_type'], you know to filter with df[df['channel_type'] == 'GP']."
    ),
)
def get_term_definition(payload: GetTermDefinitionInput) -> Optional[Dict[str, Any]]:
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
    logger.info(f"INPUT - term: {payload.term}")

    index = _ensure_index_built()

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
        logger.info(f"Found similar match: {entry.term} (score: {hits[0].score:.3f})")
        logger.info("=" * 60)
        return entry.model_dump()

    logger.info("No match found")
    logger.info("=" * 60)
    return None


@analysis_mcp.tool(
    name="search_knowledge",
    description=(
        "Search the knowledge base for relevant information about terms, concepts, or topics. "
        "This is a broader search tool compared to get_term_definition - use this when you need to explore concepts "
        "or when exact term lookup fails. "
        "Searches both term definitions and document chunks, returning results ranked by similarity score. "
        "\n\n"
        "Use this tool when: "
        "- You need to understand a concept or topic (not just a specific term) "
        "- get_term_definition() didn't find an exact match "
        "- You want to explore related information about a domain topic "
        "- You need to understand how datasets are structured or what columns mean "
        "\n\n"
        "The search uses hybrid search (combining keyword and semantic search) for better results. "
        "You can limit the search scope to 'terms' only, 'docs' only, or both (default). "
        "Returns KnowledgeHit objects with kind ('term' or 'chunk'), score, and the actual content (term_entry or chunk). "
        "\n\n"
        "Example queries: "
        "- 'patient data structure' "
        "- 'MR activity metrics' "
        "- 'COVID-19 data format' "
        "- 'how to filter by channel type'"
    ),
)
def search_knowledge(payload: SearchKnowledgeInput) -> Dict[str, List[Dict[str, Any]]]:
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
    logger.info(f"INPUT - query: {payload.query}")
    logger.info(f"INPUT - scopes: {payload.scopes}")
    logger.info(f"INPUT - top_k: {payload.top_k}")

    scopes = payload.scopes if payload.scopes is not None else ["terms", "docs"]

    # Validate scopes (Pydantic should handle this, but double-check)
    valid_scopes = ["terms", "docs"]
    invalid_scopes = [s for s in scopes if s not in valid_scopes]
    if invalid_scopes:
        raise ValueError(
            f"Invalid scopes: {invalid_scopes}. Valid scopes are: {valid_scopes}"
        )

    index = _ensure_index_built()
    hits = index.search(query=payload.query, scopes=scopes, top_k=payload.top_k)

    # Convert to dictionaries
    hits_dict = [hit.model_dump() for hit in hits]

    result = {"hits": hits_dict}

    logger.info(f"OUTPUT SUMMARY: {len(hits)} results found")
    logger.debug(f"  - Results: {[(h.kind, h.score) for h in hits]}")
    logger.info("=" * 60)

    return result
