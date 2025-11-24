"""Schemas for knowledge domain."""

from .input import (
    GetDocumentMetadataInput,
    GetTermDefinitionInput,
    SearchKnowledgeInput,
)
from .output import DocumentMetadata, KnowledgeHit, TermEntry

__all__ = [
    "GetDocumentMetadataInput",
    "GetTermDefinitionInput",
    "SearchKnowledgeInput",
    "DocumentMetadata",
    "TermEntry",
    "KnowledgeHit",
]
