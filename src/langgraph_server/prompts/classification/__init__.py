"""Classification prompts for query routing."""

from .combined import COMBINED_CLASSIFICATION_PROMPT
from .doc_action import DOC_ACTION_CLASSIFICATION_PROMPT
from .query import CLASSIFICATION_PROMPT

__all__ = [
    "CLASSIFICATION_PROMPT",
    "DOC_ACTION_CLASSIFICATION_PROMPT",
    "COMBINED_CLASSIFICATION_PROMPT",
]
