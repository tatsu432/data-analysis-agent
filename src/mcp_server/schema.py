"""Pydantic schemas for MCP tool inputs and outputs."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetSchemaOutput(BaseModel):
    """Output schema for dataset schema information."""

    columns: List[str] = Field(description="List of column names")
    dtypes: Dict[str, str] = Field(description="Dictionary mapping column names to data types")
    sample_rows: List[Dict[str, Any]] = Field(description="First 5 rows as JSON")
    row_count: int = Field(description="Total number of rows")
    description: str = Field(description="Description of the dataset")


class AnalysisResultOutput(BaseModel):
    """Output schema for analysis execution results."""

    stdout: str = Field(description="Captured standard output")
    error: Optional[str] = Field(default=None, description="Error or warning messages")
    result_df_preview: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Preview of result_df (first 10 rows)"
    )
    result_df_row_count: Optional[int] = Field(
        default=None, description="Number of rows in result_df"
    )
    plot_valid: Optional[bool] = Field(
        default=None, description="Boolean indicating if plot contains data"
    )
    plot_validation_message: Optional[str] = Field(
        default=None, description="Message about plot validation"
    )
    plot_path: Optional[str] = Field(
        default=None, description="Absolute path to the saved plot file (if plot was created)"
    )
    success: bool = Field(description="Boolean indicating if execution succeeded")


class RunAnalysisInput(BaseModel):
    """Input schema for running analysis."""

    code: str = Field(
        description="Python code string to execute. The dataset is available as `df`. "
        "Date columns are automatically converted to datetime. "
        "Use pandas, numpy, and matplotlib.pyplot. Assign results to `result_df` "
        "and save plots using plt.savefig(plot_filename) where plot_filename is "
        "a variable provided in the execution environment."
    )


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

