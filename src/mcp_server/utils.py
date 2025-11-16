"""Shared utility functions for MCP server."""

import pandas as pd


def detect_and_convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect and convert time series columns to datetime.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with datetime columns converted
    """
    df = df.copy()

    # Common date column names
    date_column_names = [
        "Date",
        "date",
        "DATE",
        "time",
        "Time",
        "TIME",
        "timestamp",
        "Timestamp",
        "TIMESTAMP",
    ]

    # Try to convert known date column names
    for col in df.columns:
        if col in date_column_names:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    # Also try to detect date-like columns by name patterns
    for col in df.columns:
        if col not in date_column_names:
            # Check if column name contains date-related keywords
            col_lower = col.lower()
            if any(
                keyword in col_lower
                for keyword in ["date", "time", "day", "month", "year"]
            ):
                try:
                    # Try to convert, but don't fail if it doesn't work
                    converted = pd.to_datetime(df[col], errors="coerce")
                    # Only convert if most values are successfully parsed
                    if converted.notna().sum() > len(df) * 0.5:
                        df[col] = converted
                except Exception:
                    pass

    return df

