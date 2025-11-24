"""Pydantic schemas for MCP tool inputs and outputs."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetSchemaOutput(BaseModel):
    """Output schema for dataset schema information."""

    columns: List[str] = Field(description="List of column names")
    dtypes: Dict[str, str] = Field(
        description="Dictionary mapping column names to data types"
    )
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
        default=None,
        description="Absolute path to the saved plot file (if plot was created)",
    )
    success: bool = Field(description="Boolean indicating if execution succeeded")


class GetDatasetSchemaInput(BaseModel):
    """Input schema for getting dataset schema information."""

    dataset_id: str = Field(
        ...,
        description=(
            "The unique identifier of the dataset to get schema for. "
            "Use list_datasets() first to see all available dataset IDs. "
            "Example values: 'jpm_patient_data', 'covid_new_cases_daily', 'mr_activity_data'"
        ),
    )


class RunAnalysisInput(BaseModel):
    """Input schema for running data analysis code."""

    code: str = Field(
        ...,
        description=(
            "Python code string to execute for data analysis. "
            "CRITICAL: This is the PRIMARY tool for performing actual data analysis. "
            "You MUST call this tool to complete any data analysis task. "
            "Do NOT stop after calling list_datasets or get_dataset_schema - those are intermediate steps. "
            "\n\n"
            "Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), sklearn, statsmodels (sm), torch, Prophet, pmdarima (pm), arch. "
            "Date columns are AUTOMATICALLY converted to datetime - you don't need to do this manually. "
            "\n\n"
            "Dataset access: "
            "- All datasets are available via dfs[dataset_id] dictionary. "
            "- Each dataset also has a code_name alias (e.g., df_covid_daily, df_jpm_patients, df_mr_activity). "
            "- If primary_dataset_id is provided, that dataset is also available as 'df'. "
            "- If only one dataset is provided, it's automatically available as 'df'. "
            "\n\n"
            "Output requirements: "
            "- Assign your final result DataFrame to 'result_df' variable. "
            "- To create plots, use plt.savefig(plot_filename) where plot_filename is provided in the execution environment. "
            "- The plot will be automatically validated and saved. "
            "\n\n"
            "Example code structure: "
            "```python\n"
            "# Load data\n"
            "df = dfs['covid_new_cases_daily']  # or use alias df_covid_daily\n"
            "# Filter and analyze\n"
            "filtered = df[df['prefecture'] == 'Tokyo']\n"
            "# Aggregate\n"
            "result_df = filtered.groupby('date').sum()\n"
            "# Plot\n"
            "plt.figure(figsize=(10, 6))\n"
            "plt.plot(result_df.index, result_df['new_cases'])\n"
            "plt.savefig(plot_filename)\n"
            "```"
        ),
    )
    dataset_ids: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of dataset IDs to load for analysis. Must contain at least one valid dataset ID. "
            "Use list_datasets() to see all available dataset IDs. "
            "All specified datasets will be loaded and available in the code execution environment. "
            "Example: ['covid_new_cases_daily'] or ['jpm_patient_data', 'mr_activity_data']"
        ),
    )
    primary_dataset_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional primary dataset ID. If provided, this dataset will be available as 'df' variable "
            "in addition to being accessible via dfs[dataset_id]. "
            "Must be one of the dataset_ids. "
            "Useful when you want convenient access to the main dataset you're analyzing. "
            "If not provided and only one dataset is in dataset_ids, that dataset automatically becomes 'df'."
        ),
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
