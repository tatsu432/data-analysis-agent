"""Utility functions for analysis domain infrastructure."""

import re
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

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


def preprocess_code_for_deprecated_styles(code: str) -> str:
    """
    Preprocess code to replace deprecated matplotlib/seaborn styles with valid alternatives.

    Args:
        code: Python code string to preprocess

    Returns:
        Preprocessed code string with deprecated styles replaced
    """
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
                return f"plt.style.use('{fallback}')"
            else:
                return f"plt.style.use('{replacement}')"
        return match.group(0)  # Return unchanged if not in mapping

    preprocessed_code = re.sub(pattern, replace_style, code)

    # Also handle cases where style might be in a variable or other contexts
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

        # Replace standalone style names in quotes
        if f"'{old_style}'" in preprocessed_code:
            preprocessed_code = preprocessed_code.replace(
                f"'{old_style}'", f"'{replacement_style}'"
            )
        if f'"{old_style}"' in preprocessed_code:
            preprocessed_code = preprocessed_code.replace(
                f'"{old_style}"', f'"{replacement_style}"'
            )

    return preprocessed_code


def get_valid_style_fallback(style_name: str) -> str:
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


def validate_plot(plot_path: Path) -> Dict[str, Any]:
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
