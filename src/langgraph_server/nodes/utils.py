"""Utility functions for nodes."""


def extract_content_text(content):
    """Extract text from content, handling both string and list formats.

    Anthropic Claude returns content as a list like [{"type": "text", "text": "..."}]
    while other models return a plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from Anthropic's content blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Anthropic format: {"type": "text", "text": "..."}
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
                # Also handle direct string values in dict
                elif "text" in item:
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    # Fallback: convert to string
    return str(content)


def is_tool_name(actual_name: str, expected_base: str) -> bool:
    """Check if a tool name matches the expected base name, handling namespace prefixes.

    When tools are imported from domain-specific servers using FastMCP.import_server(),
    they may be prefixed with the domain name (e.g., "analysis_run_analysis" instead of "run_analysis").
    This function handles both cases.

    Args:
        actual_name: The actual tool name (may be prefixed)
        expected_base: The expected base tool name (e.g., "run_analysis")

    Returns:
        True if the tool name matches (with or without prefix)
    """
    if not actual_name or not expected_base:
        return False

    # Direct match
    if actual_name == expected_base:
        return True

    # Check if it's prefixed with a domain (e.g., "analysis_run_analysis")
    # Common prefixes: "analysis_", "knowledge_", "confluence_"
    if actual_name.endswith(f"_{expected_base}"):
        return True

    # Also check if expected_base is a suffix (handles cases like "run_analysis" in "analysis_run_analysis")
    if actual_name.endswith(expected_base) and len(actual_name) > len(expected_base):
        # Make sure there's a separator (underscore) before the expected_base
        prefix = actual_name[: -len(expected_base) - 1]
        if prefix and prefix.endswith("_"):
            return True

    return False
