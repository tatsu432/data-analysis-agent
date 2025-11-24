"""Use case for running data analysis code."""

import io
import logging
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

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

import re

import arch
import matplotlib
import numpy as np
import pandas as pd
import pmdarima as pm
import sklearn
import statsmodels
import statsmodels.api as sm
import torch
from prophet import Prophet
from sklearn import (
    linear_model,
    metrics,
    model_selection,
    preprocessing,
)

from ..infrastructure.dataset_store import DatasetStore
from ..infrastructure.datasets_registry import DATASETS
from ..infrastructure.utils import (
    detect_and_convert_datetime_columns,
    get_valid_style_fallback,
    preprocess_code_for_deprecated_styles,
    validate_plot,
)
from ..schema.input import RunAnalysisInput
from ..schema.output import AnalysisResultOutput

logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402 - Must import after setting backend


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


class RunAnalysisUseCase:
    """Use case for executing Python code for data analysis."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.dataset_store = DatasetStore(DATASETS)
        self.datasets = DATASETS

    def execute(self, payload: RunAnalysisInput) -> AnalysisResultOutput:
        """
        Executes Python code in a controlled environment with one or more datasets loaded.

        Args:
            payload: Input containing code, dataset_ids, and optional primary_dataset_id

        Returns:
            AnalysisResultOutput containing execution results

        Raises:
            ValueError: If dataset_ids is empty, contains invalid IDs, or primary_dataset_id is invalid
        """
        logger.info("=" * 60)
        logger.info("USECASE EXECUTION: run_analysis")
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
        invalid_ids = [
            ds_id for ds_id in payload.dataset_ids if ds_id not in self.datasets
        ]
        if invalid_ids:
            available_ids = ", ".join(self.datasets.keys())
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
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent

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
            df = self.dataset_store.load_dataframe(ds_id)

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
        for ds_id, meta in self.datasets.items():
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
            logger.info(
                f"Single dataset '{payload.dataset_ids[0]}' bound to variable 'df'"
            )

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
            preprocessed_code = preprocess_code_for_deprecated_styles(payload.code)
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
                    style_match = re.search(r"'([^']+)' is not a valid", error_str)
                    if style_match:
                        invalid_style = style_match.group(1)
                        fallback_style = get_valid_style_fallback(invalid_style)
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
                logger.info(
                    f"Plot file found at fallback location: {plot_path_fallback}"
                )
                # Move it to the correct location for consistency
                try:
                    found_plot_path.rename(plot_path)
                    found_plot_path = plot_path
                    logger.info(f"Moved plot to correct location: {plot_path}")
                except Exception as e:
                    logger.warning(f"Could not move plot to img directory: {e}")

            if found_plot_path and found_plot_path.exists():
                # Validate the plot
                plot_validation = validate_plot(found_plot_path)
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
