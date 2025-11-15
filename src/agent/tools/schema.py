"""Tool to get dataset schema information."""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Import the datetime conversion function for consistency
from .execution import _detect_and_convert_datetime_columns

logger = logging.getLogger(__name__)


def get_dataset_schema() -> Dict[str, Any]:
    """
    Returns schema information about the COVID-19 dataset.

    Returns:
        Dictionary containing:
        - columns: list of column names
        - dtypes: dictionary mapping column names to data types
        - sample_rows: first 5 rows as JSON
        - row_count: total number of rows
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: get_dataset_schema")
    logger.info("=" * 60)
    logger.info("INPUT - No parameters required")

    # Get the path to the CSV file
    project_root = Path(__file__).parent.parent.parent.parent
    csv_path = project_root / "data" / "newly_confirmed_cases_daily.csv"

    # Load the dataset
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    # Use the same datetime conversion function as execution tool for consistency
    df = _detect_and_convert_datetime_columns(df)

    # Get column information
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Get sample rows (first 5)
    sample_df = df.head(5)
    sample_rows = sample_df.to_dict(orient="records")

    # Convert date-like columns and NaN values to strings/None for JSON serialization
    for row in sample_rows:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                row[key] = str(value)
            elif isinstance(value, (int, float)) and pd.isna(value):
                row[key] = None

    result = {
        "columns": columns,
        "dtypes": dtypes,
        "sample_rows": sample_rows,
        "row_count": len(df),
        "description": "COVID-19 newly confirmed cases daily data for Japanese prefectures",
    }

    logger.info("OUTPUT SUMMARY:")
    logger.info(f"  - Columns: {len(columns)}")
    logger.info(f"  - Row count: {len(df)}")
    logger.info(f"  - Sample rows: {len(sample_rows)}")
    logger.debug(f"  - Column names: {columns}")
    logger.info("=" * 60)

    return result
