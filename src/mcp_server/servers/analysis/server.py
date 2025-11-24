"""FastMCP server for analysis domain tools.

This module provides MCP tools for data analysis:
- list_datasets: List available datasets
- get_dataset_schema: Get dataset schema information
- run_analysis: Execute Python code for data analysis
"""

import logging

from fastmcp import FastMCP

from .schema.input import GetDatasetSchemaInput, RunAnalysisInput
from .schema.output import AnalysisResultOutput, DatasetSchemaOutput
from .usecases.get_dataset_schema_usecase import GetDatasetSchemaUseCase
from .usecases.list_datasets_usecase import ListDatasetsUseCase
from .usecases.run_analysis_usecase import RunAnalysisUseCase

logger = logging.getLogger(__name__)

# Create FastMCP instance for analysis domain
analysis_mcp = FastMCP("Analysis Tools")

# Initialize use cases
_list_datasets_usecase = ListDatasetsUseCase()
_get_dataset_schema_usecase = GetDatasetSchemaUseCase()
_run_analysis_usecase = RunAnalysisUseCase()


@analysis_mcp.tool(
    name="list_datasets",
    description=(
        "List all available datasets for analysis. "
        "This is an INTERMEDIATE step - use this to discover available datasets, "
        "then you MUST continue to get_dataset_schema() and run_analysis() to complete the task. "
        "Returns dataset IDs, descriptions, code aliases, and storage information. "
        "Available datasets include: jpm_patient_data, jamdas_patient_data, covid_new_cases_daily, mr_activity_data. "
        "CRITICAL: Do NOT stop after calling this tool - it only provides information, not analysis results."
    ),
)
def list_datasets() -> dict:
    """List all available datasets."""
    return _list_datasets_usecase.execute()


@analysis_mcp.tool(
    name="get_dataset_schema",
    description=(
        "Get detailed schema information for a specific dataset. "
        "This is an INTERMEDIATE step - use this to understand dataset structure, "
        "then you MUST continue to run_analysis() to complete the task. "
        "Returns column names, data types, sample rows (first 5), total row count, and dataset description. "
        "Use this to understand what columns are available, their types, and see example data before writing analysis code. "
        "CRITICAL: Do NOT stop after calling this tool - it only provides information, not analysis results. "
        "You must call run_analysis() to perform actual data analysis."
    ),
)
def get_dataset_schema(payload: GetDatasetSchemaInput) -> DatasetSchemaOutput:
    """Get schema information for a dataset."""
    return _get_dataset_schema_usecase.execute(payload)


@analysis_mcp.tool(
    name="run_analysis",
    description=(
        "CRITICAL: This is the PRIMARY tool for performing actual data analysis. "
        "You MUST call this tool to complete any data analysis task. "
        "Do NOT stop after calling list_datasets or get_dataset_schema - those are intermediate steps. "
        "\n\n"
        "Execute Python code for data analysis on one or more datasets. "
        "This tool executes your analysis code and returns actual results including data previews, plots, and execution status. "
        "This is the ONLY tool that produces analysis results - all other tools (list_datasets, get_dataset_schema) are for information gathering only. "
        "\n\n"
        "Dataset Access: "
        "- All datasets are available via dfs[dataset_id] dictionary. "
        "- Each dataset also has a code_name alias (e.g., df_covid_daily, df_jpm_patients, df_mr_activity). "
        "- If primary_dataset_id is provided, that dataset is also available as 'df'. "
        "- If only one dataset is provided, it's automatically available as 'df'. "
        "\n\n"
        "Available Libraries: "
        "- pandas (pd), numpy (np), matplotlib.pyplot (plt) "
        "- sklearn: linear_model, metrics, model_selection, preprocessing "
        "- statsmodels (sm): OLS, GLM, ARIMA, SARIMAX, VAR, etc. "
        "- torch: PyTorch for deep learning "
        "- Prophet: Facebook Prophet for time series forecasting "
        "- pmdarima (pm): Auto ARIMA "
        "- arch: ARCH/GARCH models for volatility "
        "\n\n"
        "Output Requirements: "
        "- Assign your final result DataFrame to 'result_df' variable. "
        "- To create plots, use plt.savefig(plot_filename) where plot_filename is provided in the execution environment. "
        "- Date columns are AUTOMATICALLY converted to datetime - you don't need to do this manually. "
        "\n\n"
        "The tool returns: "
        "- result_df_preview: First 10 rows of your result_df "
        "- result_df_row_count: Number of rows (check if 0 - means no data matched your filters) "
        "- plot_path: Path to saved plot file (if plot was created) "
        "- plot_valid: Whether the plot contains actual data "
        "- error: Any errors or warnings "
        "- success: Whether execution succeeded "
        "\n\n"
        "IMPORTANT: Always check result_df_row_count - if it's 0, your filtering returned no data. "
        "Check date formats, column names, and filter conditions."
    ),
)
def run_analysis(payload: RunAnalysisInput) -> AnalysisResultOutput:
    """Execute Python code for data analysis."""
    return _run_analysis_usecase.execute(payload)
