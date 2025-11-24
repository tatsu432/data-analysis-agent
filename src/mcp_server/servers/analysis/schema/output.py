"""Output schemas for analysis domain."""

from typing import Any, Dict, List, Optional

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
