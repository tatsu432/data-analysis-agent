"""Tool to execute Python code for COVID-19 data analysis."""

import io
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402 - Must import after setting backend

# Set ggplot style for all plots
plt.style.use("ggplot")


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
    - Plot filename is available as `plot_filename` variable (e.g., 'plot_20251115_212901.png')
      Use plt.savefig(plot_filename) to save plots

    Args:
        code: Python code string to execute

    Returns:
        Dictionary containing:
        - stdout: captured standard output
        - error: error message if execution failed
        - result_df_preview: preview of result_df if defined (first 10 rows)
        - result_df_row_count: number of rows in result_df if defined
        - plot_valid: boolean indicating if plot is valid and contains data
        - plot_validation_message: message about plot validation
        - plot_path: absolute path to the saved plot file (if plot was created)
        - success: boolean indicating if execution succeeded

        Note: The plot file is saved to the project root directory and kept for user access.
        The UI layer can load the plot directly from plot_path. Base64 encoding is not used
        to save tokens and computation.
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: run_covid_analysis")
    logger.info("=" * 60)
    logger.info(f"INPUT - Code length: {len(code)} characters")
    logger.debug(f"INPUT - Code content:\n{code}")

    # Generate timestamp for plot filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plot_{timestamp}.png"
    logger.info(f"Plot filename for this execution: {plot_filename}")

    # Get the path to the CSV file
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "data" / "newly_confirmed_cases_daily.csv"

    # Load the dataset
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    # Automatically detect and convert datetime columns
    df = _detect_and_convert_datetime_columns(df)

    # Prepare execution environment
    # Make plot filename available to the executed code
    exec_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df,
        "plot_filename": plot_filename,  # Available for use in code
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
        "plot_valid": None,
        "plot_validation_message": None,
        "plot_path": None,
        "success": False,
    }

    try:
        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute the code
        logger.info("Executing user code...")
        exec(code, exec_globals)
        logger.info("Code execution completed successfully")

        # Check if result_df was created
        if "result_df" in exec_globals:
            result_df = exec_globals["result_df"]
            if isinstance(result_df, pd.DataFrame):
                # Get row count
                result["result_df_row_count"] = len(result_df)
                logger.info(
                    f"result_df created: {len(result_df)} rows, {len(result_df.columns)} columns"
                )

                # Get preview (first 10 rows)
                preview = result_df.head(10)
                result["result_df_preview"] = preview.to_dict(orient="records")

                # Warn if result_df is empty
                if len(result_df) == 0:
                    warning_msg = (
                        "WARNING: result_df is empty (0 rows). "
                        "This usually means your filtering conditions returned no data. "
                        "Please check: "
                        "1. Date format matches the dataset (dates are automatically converted to datetime), "
                        "2. Column names are correct, "
                        "3. Filter conditions are not too restrictive."
                    )
                    result["error"] = warning_msg
                    logger.warning(warning_msg)

        # Check if plot was saved (using timestamped filename)
        plot_path = project_root / plot_filename
        if plot_path.exists():
            logger.info(f"Plot file found: {plot_path}")
            # Validate the plot
            plot_validation = _validate_plot(plot_path)
            result["plot_valid"] = plot_validation["is_valid"]
            result["plot_validation_message"] = plot_validation["error_message"]

            if plot_validation["is_valid"]:
                logger.info("Plot validation: PASSED")
            else:
                logger.warning(
                    f"Plot validation: FAILED - {plot_validation['error_message']}"
                )

            # Return plot path (base64 encoding removed to save tokens - UI can load from file)
            result["plot_path"] = str(plot_path)
            logger.info(
                f"Plot saved at: {plot_path} (file size: {plot_path.stat().st_size} bytes)"
            )

            # If plot is invalid, add warning to error message
            if not plot_validation["is_valid"]:
                error_msg = plot_validation["error_message"]
                if result["error"]:
                    result["error"] = (
                        f"{result['error']}\n\nPLOT VALIDATION ERROR: {error_msg}"
                    )
                else:
                    result["error"] = f"PLOT VALIDATION ERROR: {error_msg}"
        else:
            result["plot_valid"] = False
            result["plot_validation_message"] = "No plot file was created."
            result["plot_path"] = None
            logger.info("No plot file was created")

        result["success"] = True
        logger.info("Execution completed successfully")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result["error"] = error_msg
        result["success"] = False
        logger.error(f"Execution failed: {type(e).__name__}: {str(e)}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")

    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Get captured output
        result["stdout"] = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        if stderr_output and not result["error"]:
            result["error"] = stderr_output

        # Log output summary
        logger.info("OUTPUT SUMMARY:")
        logger.info(f"  - Success: {result['success']}")
        logger.info(f"  - stdout length: {len(result['stdout'])} characters")
        logger.info(f"  - Has error: {result['error'] is not None}")
        logger.info(
            f"  - result_df_row_count: {result.get('result_df_row_count', 'N/A')}"
        )
        logger.info(f"  - plot_valid: {result.get('plot_valid', 'N/A')}")
        if result.get("stdout"):
            logger.debug(f"  - stdout content:\n{result['stdout']}")
        if result.get("error"):
            logger.debug(f"  - error content:\n{result['error']}")
        logger.info("=" * 60)

    return result
