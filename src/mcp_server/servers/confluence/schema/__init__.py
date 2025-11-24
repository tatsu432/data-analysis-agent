"""Schemas for Confluence domain."""

from .input import (
    ConfluenceCreatePageInput,
    ConfluenceGetPageInput,
    ConfluenceSearchPagesInput,
    ConfluenceUpdatePageInput,
)
from .output import (
    ConfluenceCreatePageOutput,
    ConfluenceGetPageOutput,
    ConfluenceSearchPagesOutput,
    ConfluenceUpdatePageOutput,
)

__all__ = [
    "ConfluenceSearchPagesInput",
    "ConfluenceGetPageInput",
    "ConfluenceCreatePageInput",
    "ConfluenceUpdatePageInput",
    "ConfluenceSearchPagesOutput",
    "ConfluenceGetPageOutput",
    "ConfluenceCreatePageOutput",
    "ConfluenceUpdatePageOutput",
]
