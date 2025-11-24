"""Input schemas for analysis domain."""

from typing import List, Optional

from pydantic import BaseModel, Field


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
