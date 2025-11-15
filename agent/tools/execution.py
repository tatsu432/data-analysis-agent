"""Tool to execute Python code for COVID-19 data analysis."""

import base64
import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


def _detect_and_convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _validate_plot(plot_path: Path) -> Dict[str, Any]:
    """
    Validate that a plot file exists and contains actual data.

    Args:
        plot_path: Path to the plot file

    Returns:
        Dictionary with validation results:
        - is_valid: boolean indicating if plot is valid
        - is_empty: boolean indicating if plot appears empty
        - error_message: error message if validation failed
    """
    if not plot_path.exists():
        return {
            "is_valid": False,
            "is_empty": True,
            "error_message": "Plot file was not created.",
        }

    try:
        # Load the image
        img = Image.open(plot_path)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get image dimensions
        width, height = img.size

        # Check if image is too small (likely empty or error)
        if width < 100 or height < 100:
            return {
                "is_valid": False,
                "is_empty": True,
                "error_message": f"Plot image is too small ({width}x{height}), likely empty or invalid.",
            }

        # Convert to numpy array for analysis
        img_array = np.array(img)

        # Check if image is mostly white/empty (common for empty plots)
        # Calculate the percentage of white pixels
        white_threshold = 250  # RGB values above this are considered "white"
        white_pixels = np.sum(np.all(img_array >= white_threshold, axis=2))
        total_pixels = width * height
        white_percentage = white_pixels / total_pixels

        # If more than 90% of the image is white, it's likely empty
        if white_percentage > 0.90:
            return {
                "is_valid": False,
                "is_empty": True,
                "error_message": f"Plot appears to be empty ({(white_percentage * 100):.1f}% white space). "
                f"This usually means no data was plotted. Check your filtering conditions.",
            }

        # Check if image has very low variance (likely a blank plot)
        gray = np.mean(img_array, axis=2)
        variance = np.var(gray)
        if variance < 10:  # Very low variance suggests empty plot
            return {
                "is_valid": False,
                "is_empty": True,
                "error_message": f"Plot has very low variance ({variance:.2f}), likely empty. "
                f"Check that your data filtering returned results.",
            }

        return {"is_valid": True, "is_empty": False, "error_message": None}

    except Exception as e:
        return {
            "is_valid": False,
            "is_empty": True,
            "error_message": f"Error validating plot: {str(e)}",
        }


def run_covid_analysis(code: str) -> Dict[str, Any]:
    """
    Executes Python code in a controlled environment with the COVID dataset loaded.

    Execution rules:
    - The CSV is loaded as `df`
    - Allowed imports: pandas, numpy, matplotlib.pyplot
    - Captures stdout, errors, dataframe preview, and plots
    - Automatically converts time series columns to datetime

    Args:
        code: Python code string to execute

    Returns:
        Dictionary containing:
        - stdout: captured standard output
        - error: error message if execution failed
        - result_df_preview: preview of result_df if defined (first 10 rows)
        - result_df_row_count: number of rows in result_df if defined
        - plot_base64: base64-encoded plot if analysis_plot.png was created
        - plot_valid: boolean indicating if plot is valid and contains data
        - plot_validation_message: message about plot validation
        - success: boolean indicating if execution succeeded
    """
    # Get the path to the CSV file
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "data" / "newly_confirmed_cases_daily.csv"

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Automatically detect and convert datetime columns
    df = _detect_and_convert_datetime_columns(df)

    # Prepare execution environment
    exec_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df,
        "__builtins__": __builtins__,
    }

    # Capture stdout
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = {
        "stdout": "",
        "error": None,
        "result_df_preview": None,
        "result_df_row_count": None,
        "plot_base64": None,
        "plot_valid": None,
        "plot_validation_message": None,
        "success": False,
    }

    try:
        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute the code
        exec(code, exec_globals)

        # Check if result_df was created
        if "result_df" in exec_globals:
            result_df = exec_globals["result_df"]
            if isinstance(result_df, pd.DataFrame):
                # Get row count
                result["result_df_row_count"] = len(result_df)

                # Get preview (first 10 rows)
                preview = result_df.head(10)
                result["result_df_preview"] = preview.to_dict(orient="records")

                # Warn if result_df is empty
                if len(result_df) == 0:
                    result["error"] = (
                        "WARNING: result_df is empty (0 rows). "
                        "This usually means your filtering conditions returned no data. "
                        "Please check: "
                        "1. Date format matches the dataset (dates are automatically converted to datetime), "
                        "2. Column names are correct, "
                        "3. Filter conditions are not too restrictive."
                    )

        # Check if plot was saved
        plot_path = project_root / "analysis_plot.png"
        if plot_path.exists():
            # Validate the plot before encoding
            plot_validation = _validate_plot(plot_path)
            result["plot_valid"] = plot_validation["is_valid"]
            result["plot_validation_message"] = plot_validation["error_message"]

            # Read and encode the plot
            with open(plot_path, "rb") as f:
                plot_bytes = f.read()
                result["plot_base64"] = base64.b64encode(plot_bytes).decode("utf-8")

            # If plot is invalid, add warning to error message
            if not plot_validation["is_valid"]:
                error_msg = plot_validation["error_message"]
                if result["error"]:
                    result["error"] = (
                        f"{result['error']}\n\nPLOT VALIDATION ERROR: {error_msg}"
                    )
                else:
                    result["error"] = f"PLOT VALIDATION ERROR: {error_msg}"

            # Clean up the plot file
            plot_path.unlink()
        else:
            result["plot_valid"] = False
            result["plot_validation_message"] = "No plot file was created."

        result["success"] = True

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result["error"] = error_msg
        result["success"] = False

    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Get captured output
        result["stdout"] = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        if stderr_output and not result["error"]:
            result["error"] = stderr_output

    return result
