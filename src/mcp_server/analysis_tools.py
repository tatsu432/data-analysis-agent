"""MCP tools for data analysis."""

import io
import logging
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Suppress plotly import warning from pandas (we use matplotlib, not plotly)
warnings.filterwarnings(
    "ignore",
    message=".*plotly.*",
    category=UserWarning,
)

# Suppress matplotlib font warnings for missing glyphs (we handle Japanese fonts separately)
warnings.filterwarnings(
    "ignore",
    message=".*Glyph.*missing from font.*",
    category=UserWarning,
    module="matplotlib",
)

import arch
import matplotlib
import numpy as np
import pandas as pd
import pmdarima as pm
import sklearn
import statsmodels
import statsmodels.api as sm
import torch
from fastmcp import FastMCP
from PIL import Image
from prophet import Prophet
from sklearn import (
    linear_model,
    metrics,
    model_selection,
    preprocessing,
)

from .dataset_store import DatasetStore
from .datasets_registry import DATASETS
from .schema import (
    AnalysisResultOutput,
    DatasetSchemaOutput,
    GetDatasetSchemaInput,
    RunAnalysisInput,
)
from .utils import detect_and_convert_datetime_columns

logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402 - Must import after setting backend


# Configure matplotlib to support Japanese characters
def _configure_japanese_font():
    """Configure matplotlib to use a font that supports Japanese characters."""
    # Try to find a Japanese-compatible font
    # Priority: Noto Sans CJK > Hiragino Sans (macOS) > MS Gothic (Windows) > fallback
    japanese_fonts = [
        "Noto Sans CJK JP",  # Google Noto Sans CJK (if installed)
        "Noto Sans CJK SC",  # Simplified Chinese variant (also supports Japanese)
        "Hiragino Sans",  # macOS default Japanese font
        "Hiragino Kaku Gothic ProN",  # macOS alternative
        "Hiragino Kaku Gothic Pro",  # macOS alternative (without N)
        "Yu Gothic",  # Windows 10+ Japanese font
        "MS Gothic",  # Windows Japanese font
        "Takao Gothic",  # Linux Japanese font
    ]

    # Get list of available fonts (case-insensitive search)
    available_fonts = {f.name.lower(): f.name for f in fm.fontManager.ttflist}
    available_font_names = list(available_fonts.values())

    # Try to find a Japanese-compatible font (case-insensitive)
    selected_font = None
    for font_name in japanese_fonts:
        # Try exact match first
        if font_name in available_font_names:
            selected_font = font_name
            logger.info(f"Using Japanese-compatible font: {font_name}")
            break
        # Try case-insensitive match
        font_lower = font_name.lower()
        if font_lower in available_fonts:
            selected_font = available_fonts[font_lower]
            logger.info(
                f"Using Japanese-compatible font: {selected_font} (matched {font_name})"
            )
            break
        # Try partial match (for fonts with variations)
        for available_font in available_font_names:
            if (
                font_name.lower() in available_font.lower()
                or available_font.lower() in font_name.lower()
            ):
                selected_font = available_font
                logger.info(
                    f"Using Japanese-compatible font: {selected_font} (partial match for {font_name})"
                )
                break
        if selected_font:
            break

    if selected_font:
        # Configure matplotlib to use the font
        plt.rcParams["font.family"] = selected_font
        # Also set sans-serif fallback with the Japanese font first
        current_sans_serif = plt.rcParams.get("font.sans-serif", [])
        # Remove the selected font if it's already in the list to avoid duplicates
        current_sans_serif = [f for f in current_sans_serif if f != selected_font]
        plt.rcParams["font.sans-serif"] = [selected_font] + current_sans_serif
        logger.info(f"Configured matplotlib to use font: {selected_font}")
    else:
        # If no Japanese font found, log available fonts for debugging
        logger.warning(
            "No Japanese-compatible font found. Japanese characters may not display correctly. "
            "Consider installing a font like 'Noto Sans CJK JP' or 'Hiragino Sans'."
        )
        logger.debug(f"Available fonts (first 20): {available_font_names[:20]}")


# Configure Japanese font support
_configure_japanese_font()

# Set ggplot style for all plots
plt.style.use("ggplot")

# Create FastMCP instance
analysis_mcp = FastMCP("data_analysis_mcp")

# Initialize dataset store
dataset_store = DatasetStore(DATASETS)

# Mapping of deprecated seaborn styles to valid alternatives
DEPRECATED_STYLE_MAPPING = {
    "seaborn-whitegrid": "seaborn-v0_8-whitegrid",
    "seaborn-darkgrid": "seaborn-v0_8-darkgrid",
    "seaborn-dark": "seaborn-v0_8-dark",
    "seaborn-white": "seaborn-v0_8-white",
    "seaborn-ticks": "seaborn-v0_8-ticks",
    "seaborn-colorblind": "seaborn-v0_8-colorblind",
    "seaborn-pastel": "seaborn-v0_8-pastel",
    "seaborn-paper": "seaborn-v0_8-paper",
    "seaborn-notebook": "seaborn-v0_8-notebook",
    "seaborn-talk": "seaborn-v0_8-talk",
    "seaborn-poster": "seaborn-v0_8-poster",
}


def _preprocess_code_for_deprecated_styles(code: str) -> str:
    """
    Preprocess code to replace deprecated matplotlib/seaborn styles with valid alternatives.

    Args:
        code: Python code string to preprocess

    Returns:
        Preprocessed code string with deprecated styles replaced
    """
    import re

    # Get available styles to check if replacements are valid
    try:
        available_styles = set(plt.style.available)
    except Exception:
        available_styles = set()

    # Pattern to match plt.style.use() calls with string arguments
    # Matches: plt.style.use('style-name') or plt.style.use("style-name")
    pattern = r"plt\.style\.use\(['\"]([^'\"]+)['\"]\)"

    def replace_style(match):
        style_name = match.group(1)
        if style_name in DEPRECATED_STYLE_MAPPING:
            replacement = DEPRECATED_STYLE_MAPPING[style_name]
            # Check if replacement style is available, otherwise use a safe fallback
            if replacement not in available_styles:
                # Use a safe fallback based on the style type
                if "whitegrid" in style_name.lower() or "white" in style_name.lower():
                    fallback = "default"
                elif "dark" in style_name.lower():
                    fallback = "dark_background"
                else:
                    fallback = "default"
                logger.info(
                    f"Replacing deprecated style '{style_name}' with fallback '{fallback}' "
                    f"(replacement '{replacement}' not available)"
                )
                return f"plt.style.use('{fallback}')"
            else:
                logger.info(
                    f"Replacing deprecated style '{style_name}' with '{replacement}'"
                )
                return f"plt.style.use('{replacement}')"
        return match.group(0)  # Return unchanged if not in mapping

    preprocessed_code = re.sub(pattern, replace_style, code)

    # Also handle cases where style might be in a variable or other contexts
    # This is a more general replacement for common deprecated style names
    for old_style, new_style in DEPRECATED_STYLE_MAPPING.items():
        # Check if replacement is available
        if new_style not in available_styles:
            # Use fallback
            if "whitegrid" in old_style.lower() or "white" in old_style.lower():
                fallback = "default"
            elif "dark" in old_style.lower():
                fallback = "dark_background"
            else:
                fallback = "default"
            replacement_style = fallback
        else:
            replacement_style = new_style

        # Replace standalone style names in quotes (but be careful not to break strings)
        # Only replace if it's clearly a style name argument
        if f"'{old_style}'" in preprocessed_code:
            preprocessed_code = preprocessed_code.replace(
                f"'{old_style}'", f"'{replacement_style}'"
            )
        if f'"{old_style}"' in preprocessed_code:
            preprocessed_code = preprocessed_code.replace(
                f'"{old_style}"', f'"{replacement_style}"'
            )

    return preprocessed_code


def _get_valid_style_fallback(style_name: str) -> str:
    """
    Get a valid matplotlib style as a fallback for invalid styles.

    Args:
        style_name: The invalid style name that was attempted

    Returns:
        A valid style name to use as fallback
    """
    # Try to find a similar valid style
    if "whitegrid" in style_name.lower() or "white" in style_name.lower():
        # Try seaborn-v0_8 styles first, then fall back to built-in
        try:
            if "seaborn-v0_8-whitegrid" in plt.style.available:
                return "seaborn-v0_8-whitegrid"
        except Exception:
            pass
        return "default"  # Fall back to default style
    elif "dark" in style_name.lower():
        try:
            if "seaborn-v0_8-darkgrid" in plt.style.available:
                return "seaborn-v0_8-darkgrid"
        except Exception:
            pass
        return "dark_background"
    else:
        # Default fallback
        return "default"


# _detect_and_convert_datetime_columns moved to utils.py
# Imported as detect_and_convert_datetime_columns for backward compatibility
_detect_and_convert_datetime_columns = detect_and_convert_datetime_columns


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
def list_datasets() -> Dict[str, Any]:
    """
    Returns a list of available datasets with their metadata.

    Returns:
        Dictionary with a "datasets" key containing list of dataset metadata
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: list_datasets")
    logger.info("=" * 60)

    datasets_list = []
    for ds_id, meta in DATASETS.items():
        # Get storage information
        storage = meta.get("storage")
        if storage:
            storage_kind = storage.get("kind", "unknown")

            # Build location_hint based on storage kind
            if storage_kind == "local_csv":
                path = storage.get("path")
                location_hint = str(path) if path else "local"
            elif storage_kind == "s3_csv":
                bucket = storage.get("bucket", "")
                key = storage.get("key", "")
                # Truncate long keys for display
                if len(key) > 50:
                    key_display = key[:47] + "..."
                else:
                    key_display = key
                location_hint = (
                    f"s3://{bucket}/{key_display}" if bucket and key else "s3"
                )
            else:
                location_hint = "unknown"
        else:
            # Backward compatibility: use path if available
            storage_kind = "local_csv"  # Assume local if no storage block
            location_hint = str(meta.get("path", "unknown"))

        dataset_info = {
            "id": ds_id,
            "description": meta.get("description", ""),
            "code_name": meta.get("code_name"),
            "storage_kind": storage_kind,
            "location_hint": location_hint,
        }
        datasets_list.append(dataset_info)

    result = {"datasets": datasets_list}

    logger.info(f"OUTPUT SUMMARY: {len(datasets_list)} datasets available")
    logger.debug(f"  - Dataset IDs: {[ds['id'] for ds in datasets_list]}")
    logger.info("=" * 60)

    return result


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
    """
    Returns schema information for the specified dataset.

    Args:
        dataset_id: ID of the dataset to get schema for

    Returns:
        DatasetSchemaOutput with columns, dtypes, sample rows, and row count

    Raises:
        ValueError: If dataset_id is not found in the registry
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: get_dataset_schema")
    logger.info("=" * 60)
    logger.info(f"INPUT - dataset_id: {payload.dataset_id}")

    # Use DatasetStore to get schema (handles validation and loading)
    try:
        result = dataset_store.get_schema(payload.dataset_id)

        logger.info("OUTPUT SUMMARY:")
        logger.info(f"  - Columns: {len(result.columns)}")
        logger.info(f"  - Row count: {result.row_count}")
        logger.info(f"  - Sample rows: {len(result.sample_rows)}")
        logger.debug(f"  - Column names: {result.columns}")
        logger.info("=" * 60)

        return result
    except ValueError as e:
        # Re-raise ValueError as-is (already has good error message)
        logger.error(str(e))
        raise
    except Exception as e:
        # Wrap other exceptions with context
        error_msg = f"Failed to get schema for dataset '{payload.dataset_id}': {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


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
    """
    Executes Python code in a controlled environment with one or more datasets loaded.

    Execution rules:
    - Datasets are loaded as DataFrames accessible via dfs[dataset_id]
    - Each dataset also has a code_name alias (e.g., df_covid_daily, df_jpm_patients)
    - If primary_dataset_id is provided, that dataset is also available as `df`
    - If only one dataset is provided, it's automatically available as `df`
    - Allowed libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), sklearn, statsmodels, torch, prophet, pmdarima, arch
      - sklearn modules available: linear_model, metrics, model_selection, preprocessing
        - You can use: sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing
        - Or import directly: from sklearn.linear_model import LinearRegression
      - statsmodels available as `sm` or `statsmodels`: comprehensive statistical modeling
        - Regression: sm.OLS, sm.GLM, sm.Logit, etc.
        - Time series: sm.tsa.ARIMA, sm.tsa.SARIMAX (seasonal ARIMA with exogenous variables)
        - State space models: sm.tsa.UnobservedComponents, sm.tsa.DynamicFactor, sm.tsa.statespace.*
          - Note: SARIMAX is built on state space framework and supports exogenous regressors
        - Vector models: sm.tsa.VAR, sm.tsa.VARMAX (vector autoregression)
        - Statistical tests: sm.stats.acorr_ljungbox, sm.stats.diagnostic, etc.
      - PyTorch available as `torch`: torch.tensor, torch.nn, torch.optim, etc.
      - Time series libraries:
        - Prophet (Facebook Prophet) available as `Prophet`: Prophet().fit(), Prophet().predict()
        - pmdarima (Auto ARIMA) available as `pm` or `pmdarima`: pm.auto_arima(), pm.ARIMA()
        - arch (ARCH/GARCH) available as `arch`: arch.arch_model(), arch.GARCH()
    - Captures stdout, errors, dataframe preview, and plots
    - Automatically converts time series columns to datetime
    - Plot filename is available as `plot_filename` variable
      Use plt.savefig(plot_filename) to save plots

    Args:
        code: Python code string to execute
        dataset_ids: List of dataset IDs to load
        primary_dataset_id: Optional primary dataset ID (must be in dataset_ids)

    Returns:
        AnalysisResultOutput containing execution results

    Raises:
        ValueError: If dataset_ids is empty, contains invalid IDs, or primary_dataset_id is invalid
    """
    logger.info("=" * 60)
    logger.info("TOOL EXECUTION: run_analysis")
    logger.info("=" * 60)
    logger.info(f"INPUT - Code length: {len(payload.code)} characters")
    logger.info(f"INPUT - dataset_ids: {payload.dataset_ids}")
    logger.info(f"INPUT - primary_dataset_id: {payload.primary_dataset_id}")
    logger.debug(f"INPUT - Code content:\n{payload.code}")

    # Validate dataset_ids
    if not payload.dataset_ids:
        error_msg = "dataset_ids cannot be empty. Use list_datasets() to see available datasets."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate all dataset_ids exist
    invalid_ids = [ds_id for ds_id in payload.dataset_ids if ds_id not in DATASETS]
    if invalid_ids:
        available_ids = ", ".join(DATASETS.keys())
        error_msg = (
            f"Invalid dataset IDs: {', '.join(invalid_ids)}. "
            f"Available datasets: {available_ids}. "
            f"Use list_datasets() to see all available datasets."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate primary_dataset_id if provided
    if payload.primary_dataset_id is not None:
        if payload.primary_dataset_id not in payload.dataset_ids:
            error_msg = (
                f"primary_dataset_id '{payload.primary_dataset_id}' must be in dataset_ids. "
                f"Provided dataset_ids: {payload.dataset_ids}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Generate timestamp for plot filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_basename = f"plot_{timestamp}.png"

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Ensure img directory exists
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Use full path to img directory so plt.savefig saves to the correct location
    plot_filename = str(img_dir / plot_basename)
    logger.info(f"Plot filename for this execution: {plot_filename}")

    # Load all datasets using DatasetStore
    dfs = {}
    for ds_id in payload.dataset_ids:
        logger.info(f"Loading dataset '{ds_id}'")
        # DatasetStore handles all storage-specific logic (local vs S3)
        df = dataset_store.load_dataframe(ds_id)

        # Automatically detect and convert datetime columns
        df = detect_and_convert_datetime_columns(df)
        dfs[ds_id] = df

    # Prepare execution environment
    exec_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sklearn": sklearn,  # Full sklearn module for imports like "from sklearn.linear_model import LinearRegression"
        "linear_model": linear_model,  # Direct access: linear_model.LinearRegression()
        "metrics": metrics,  # Direct access: metrics.mean_absolute_percentage_error()
        "model_selection": model_selection,  # Direct access: model_selection.train_test_split()
        "preprocessing": preprocessing,  # Direct access: preprocessing.StandardScaler()
        "sm": sm,  # statsmodels.api - for statistical modeling (OLS, ARIMA, etc.)
        "statsmodels": statsmodels,  # Full statsmodels module for accessing all submodules
        "torch": torch,  # PyTorch for deep learning and tensor operations
        "Prophet": Prophet,  # Facebook Prophet for time series forecasting
        "pm": pm,  # pmdarima for auto ARIMA
        "pmdarima": pm,  # Alternative name for pmdarima
        "arch": arch,  # ARCH/GARCH models for volatility modeling
        "dfs": dfs,  # Dict of all datasets
        "plot_filename": plot_filename,
        "__builtins__": __builtins__,
    }

    # Bind code_name aliases
    for ds_id, meta in DATASETS.items():
        if ds_id in dfs and "code_name" in meta:
            code_name = meta["code_name"]
            exec_globals[code_name] = dfs[ds_id]
            logger.info(f"Bound dataset '{ds_id}' to variable '{code_name}'")

    # Choose primary dataset for `df` variable
    if payload.primary_dataset_id is not None:
        exec_globals["df"] = dfs[payload.primary_dataset_id]
        logger.info(
            f"Primary dataset '{payload.primary_dataset_id}' bound to variable 'df'"
        )
    elif len(payload.dataset_ids) == 1:
        # If only one dataset, bind it to df for convenience
        exec_globals["df"] = dfs[payload.dataset_ids[0]]
        logger.info(f"Single dataset '{payload.dataset_ids[0]}' bound to variable 'df'")

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
        # Preprocess code to replace deprecated matplotlib styles
        preprocessed_code = _preprocess_code_for_deprecated_styles(payload.code)
        if preprocessed_code != payload.code:
            logger.info("Code was preprocessed to replace deprecated styles")
            logger.debug(f"Original code:\n{payload.code}")
            logger.debug(f"Preprocessed code:\n{preprocessed_code}")
            code_to_execute = preprocessed_code
        else:
            code_to_execute = payload.code

        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute the code
        logger.info("Executing user code...")
        try:
            exec(code_to_execute, exec_globals)
            logger.info("Code execution completed successfully")
        except OSError as e:
            # Catch matplotlib style errors and provide helpful error message
            error_str = str(e)
            if (
                "is not a valid package style" in error_str
                or "is not a valid" in error_str
            ):
                # Extract the style name from the error message
                import re

                style_match = re.search(r"'([^']+)' is not a valid", error_str)
                if style_match:
                    invalid_style = style_match.group(1)
                    fallback_style = _get_valid_style_fallback(invalid_style)
                    error_msg = (
                        f"Matplotlib style error: '{invalid_style}' is not a valid style. "
                        f"This style may have been deprecated in newer versions of matplotlib. "
                        f"Please use a valid style instead. "
                        f"Valid styles include: 'default', 'ggplot', 'seaborn-v0_8-whitegrid', etc. "
                        f"You can check available styles with: plt.style.available"
                    )
                    logger.error(error_msg)
                    logger.info(f"Suggested fallback style: '{fallback_style}'")
                    # Re-raise with a more helpful message
                    raise OSError(error_msg) from e
            # Re-raise if it's not a style error
            raise

        # Ensure matplotlib flushes any pending plots to disk
        plt.close("all")  # Close all figures to ensure they're flushed

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

        # Check if plot was saved
        # plot_filename is now a full path, so use it directly
        plot_path = Path(plot_filename)

        # Also check project root as fallback (in case plot was saved there)
        plot_path_fallback = project_root / plot_basename

        # Determine which path exists
        found_plot_path = None
        if plot_path.exists():
            found_plot_path = plot_path
            logger.info(f"Plot file found at expected location: {plot_path}")
        elif plot_path_fallback.exists():
            found_plot_path = plot_path_fallback
            logger.info(f"Plot file found at fallback location: {plot_path_fallback}")
            # Move it to the correct location for consistency
            try:
                found_plot_path.rename(plot_path)
                found_plot_path = plot_path
                logger.info(f"Moved plot to correct location: {plot_path}")
            except Exception as e:
                logger.warning(f"Could not move plot to img directory: {e}")

        if found_plot_path and found_plot_path.exists():
            # Validate the plot
            plot_validation = _validate_plot(found_plot_path)
            result["plot_valid"] = plot_validation["is_valid"]
            result["plot_validation_message"] = (
                plot_validation.get("error_message") or "Plot is valid."
            )

            if plot_validation["is_valid"]:
                logger.info("Plot validation: PASSED")
            else:
                logger.warning(
                    f"Plot validation: FAILED - {plot_validation.get('error_message', 'Unknown error')}"
                )

            # Return plot path (base64 encoding removed to save tokens - UI can load from file)
            result["plot_path"] = str(found_plot_path.absolute())
            logger.info(
                f"Plot saved at: {found_plot_path} (file size: {found_plot_path.stat().st_size} bytes)"
            )

            # If plot is invalid, add warning to error message
            if not plot_validation["is_valid"]:
                error_msg = plot_validation.get(
                    "error_message", "Plot validation failed"
                )
                if result["error"]:
                    result["error"] = (
                        f"{result['error']}\n\nPLOT VALIDATION ERROR: {error_msg}"
                    )
                else:
                    result["error"] = f"PLOT VALIDATION ERROR: {error_msg}"
        else:
            result["plot_valid"] = False
            result["plot_validation_message"] = (
                f"No plot file was created. Checked locations: {plot_path}, {plot_path_fallback}"
            )
            result["plot_path"] = None
            logger.warning(
                f"No plot file was created. Checked: {plot_path} and {plot_path_fallback}"
            )

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

    return AnalysisResultOutput(**result)
