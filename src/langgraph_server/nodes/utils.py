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
