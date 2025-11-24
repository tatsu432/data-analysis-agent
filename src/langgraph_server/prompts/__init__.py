"""Prompts for the data analysis agent.

This module exports all prompts organized by their role:
- Agent prompts: Main reasoning and analysis
- Classification prompts: Query routing and classification
- Code generation: Python code generation
- Document QA: Terminology questions
- Confluence: Confluence-related operations
- Knowledge enrichment: Domain term lookup
- Verifier: Response validation
"""

from . import classification, confluence
from .agent import ANALYSIS_PROMPT, SYSTEM_PROMPT

# Classification prompts
from .classification import (
    CLASSIFICATION_PROMPT,
    COMBINED_CLASSIFICATION_PROMPT,
    DOC_ACTION_CLASSIFICATION_PROMPT,
)
from .code_generation import CODE_GENERATION_PROMPT

# Confluence prompts
from .confluence import CONFLUENCE_QUERY_UNDERSTANDING_PROMPT
from .document_qa import DOCUMENT_QA_PROMPT
from .knowledge_enrichment import KNOWLEDGE_ENRICHMENT_PROMPT
from .verifier import VERIFIER_PROMPT

__all__ = [
    # Agent prompts
    "SYSTEM_PROMPT",
    "ANALYSIS_PROMPT",
    # Classification prompts
    "CLASSIFICATION_PROMPT",
    "DOC_ACTION_CLASSIFICATION_PROMPT",
    "COMBINED_CLASSIFICATION_PROMPT",
    # Code generation
    "CODE_GENERATION_PROMPT",
    # Document QA
    "DOCUMENT_QA_PROMPT",
    # Confluence
    "CONFLUENCE_QUERY_UNDERSTANDING_PROMPT",
    # Knowledge enrichment
    "KNOWLEDGE_ENRICHMENT_PROMPT",
    # Verifier
    "VERIFIER_PROMPT",
]
