"""Infrastructure layer for knowledge domain."""

from .document_store import DocumentStore
from .knowledge_index import build_global_knowledge_index
from .knowledge_index_manager import ensure_index_built
from .knowledge_registry import DOCUMENTS

__all__ = [
    "DocumentStore",
    "DOCUMENTS",
    "build_global_knowledge_index",
    "ensure_index_built",
]
