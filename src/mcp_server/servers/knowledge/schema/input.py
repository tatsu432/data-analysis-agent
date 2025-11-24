"""Input schemas for knowledge domain."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GetDocumentMetadataInput(BaseModel):
    """Input schema for getting document metadata."""

    doc_id: str = Field(
        ...,
        description=(
            "The unique identifier of the document to get metadata for. "
            "Use list_documents() first to see all available document IDs. "
            "Returns metadata including title, kind, source_path, description, and tags."
        ),
    )


class GetTermDefinitionInput(BaseModel):
    """Input schema for getting a term definition from the knowledge base."""

    term: str = Field(
        ...,
        description=(
            "The term or phrase to look up in the knowledge base. "
            "Searches for exact matches first (case-insensitive), then checks synonyms, "
            "then falls back to similarity search if no exact match is found. "
            "Returns the best matching term entry with definition, synonyms, related columns, and source information. "
            "Use this for domain-specific terms like 'GP', 'HP', 'at-risk', 'TRx', 'Rx', 'DDI', etc. "
            "Example: 'GP', 'at-risk', 'unmet medical needs'"
        ),
    )


class SearchKnowledgeInput(BaseModel):
    """Input schema for searching the knowledge base."""

    query: str = Field(
        ...,
        description=(
            "Search query string to find relevant information in the knowledge base. "
            "Searches both term definitions and document chunks. "
            "Returns results ranked by similarity score. "
            "Use this for broader searches when you need to find information about concepts, topics, or when exact term lookup fails. "
            "Example queries: 'patient data structure', 'MR activity metrics', 'COVID-19 data format'"
        ),
    )
    scopes: Optional[List[Literal["terms", "docs"]]] = Field(
        default=None,
        description=(
            "Optional list of scopes to search. "
            "Use ['terms'] to search only term definitions, "
            "['docs'] to search only document chunks, "
            "or ['terms', 'docs'] (default) to search both. "
            "Limiting scope can improve performance and relevance for specific use cases."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Maximum number of results to return per scope. "
            "Default is 5. Minimum is 1, maximum is 20. "
            "Use smaller values (3-5) for focused searches, larger values (10-20) for comprehensive exploration."
        ),
    )
