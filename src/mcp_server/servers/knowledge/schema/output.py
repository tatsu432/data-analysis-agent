"""Output schemas for knowledge domain."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document in the knowledge base."""

    doc_id: str = Field(description="Unique identifier for the document")
    title: str = Field(description="Title of the document")
    kind: Literal["excel_dictionary", "pdf_manual", "other"] = Field(
        description="Type of document"
    )
    source_path: str = Field(description="Path to the document file")
    description: Optional[str] = Field(
        default=None, description="Description of the document"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class TermEntry(BaseModel):
    """A term definition entry from a dictionary document."""

    term: str = Field(description="The term or phrase")
    definition: str = Field(description="Definition or explanation of the term")
    synonyms: List[str] = Field(
        default_factory=list, description="Alternative names or synonyms"
    )
    related_columns: List[str] = Field(
        default_factory=list, description="Dataset columns that relate to this term"
    )
    source_doc_id: str = Field(description="ID of the source document")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    extra_context: Optional[str] = Field(
        default=None, description="Additional context or notes"
    )


class DocChunk(BaseModel):
    """A chunk of text from a document."""

    chunk_id: str = Field(description="Unique identifier for the chunk")
    doc_id: str = Field(description="ID of the source document")
    text: str = Field(description="Text content of the chunk")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    section_heading: Optional[str] = Field(
        default=None, description="Section heading or title"
    )


class KnowledgeHit(BaseModel):
    """A search result from the knowledge base."""

    kind: Literal["term", "chunk"] = Field(description="Type of result")
    score: float = Field(description="Similarity score (higher is better)")
    term_entry: Optional[TermEntry] = Field(
        default=None, description="Term entry if kind is 'term'"
    )
    chunk: Optional[DocChunk] = Field(
        default=None, description="Document chunk if kind is 'chunk'"
    )
