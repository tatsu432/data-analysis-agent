"""Prompts for the data analysis agent.

This module exports all prompts organized by their role:
- Router: Intent classification
- Analysis Agent: Data analysis planning
- Knowledge Agent: Domain terminology
- Confluence Agent: Confluence operations
- Code Agent: Python code generation
- Final Responder: User-facing responses
- Verifier: Response verification
"""

from .analysis_agent import ANALYSIS_AGENT_PROMPT
from .code_agent import CODE_AGENT_PROMPT
from .confluence_agent import CONFLUENCE_AGENT_PROMPT
from .final_responder import FINAL_RESPONDER_PROMPT
from .knowledge_agent import KNOWLEDGE_AGENT_PROMPT
from .router import ROUTER_PROMPT
from .verifier import VERIFIER_PROMPT

__all__ = [
    "ROUTER_PROMPT",
    "ANALYSIS_AGENT_PROMPT",
    "KNOWLEDGE_AGENT_PROMPT",
    "CONFLUENCE_AGENT_PROMPT",
    "CODE_AGENT_PROMPT",
    "FINAL_RESPONDER_PROMPT",
    "VERIFIER_PROMPT",
]
