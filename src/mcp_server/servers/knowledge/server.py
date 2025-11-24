"""FastMCP server for knowledge domain tools.

This module provides MCP tools for knowledge base access:
- list_documents: List available knowledge documents
- get_document_metadata: Get document metadata
- get_term_definition: Get term definitions
- search_knowledge: Search knowledge base
"""

import logging

from fastmcp import FastMCP

from .schema.input import (
    GetDocumentMetadataInput,
    GetTermDefinitionInput,
    SearchKnowledgeInput,
)
from .schema.output import DocumentMetadata
from .usecases.get_document_metadata_usecase import GetDocumentMetadataUseCase
from .usecases.get_term_definition_usecase import GetTermDefinitionUseCase
from .usecases.list_documents_usecase import ListDocumentsUseCase
from .usecases.search_knowledge_usecase import SearchKnowledgeUseCase

logger = logging.getLogger(__name__)

# Create FastMCP instance for knowledge domain
knowledge_mcp = FastMCP("Knowledge Tools")

# Initialize use cases
_list_documents_usecase = ListDocumentsUseCase()
_get_document_metadata_usecase = GetDocumentMetadataUseCase()
_get_term_definition_usecase = GetTermDefinitionUseCase()
_search_knowledge_usecase = SearchKnowledgeUseCase()


@knowledge_mcp.tool(
    name="list_documents",
    description=(
        "List all available documents in the knowledge base. "
        "Returns metadata for Excel dictionaries and PDF manuals including doc_id, title, kind, source_path, description, and tags. "
        "Use this to discover what knowledge documents are available before searching or looking up terms. "
        "The knowledge base contains domain-specific terminology, definitions, and documentation that can help understand dataset columns and domain concepts."
    ),
)
def list_documents() -> dict:
    """List all available documents in the knowledge base."""
    return _list_documents_usecase.execute()


@knowledge_mcp.tool(
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
    """Get metadata for a document."""
    return _get_document_metadata_usecase.execute(payload)


@knowledge_mcp.tool(
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
def get_term_definition(payload: GetTermDefinitionInput) -> dict | None:
    """Get the definition of a term from the knowledge base."""
    return _get_term_definition_usecase.execute(payload)


@knowledge_mcp.tool(
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
def search_knowledge(payload: SearchKnowledgeInput) -> dict:
    """Search the knowledge base for relevant information."""
    return _search_knowledge_usecase.execute(payload)
