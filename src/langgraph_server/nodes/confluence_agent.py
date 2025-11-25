"""Confluence agent node with strict tool masking."""

import logging

from langchain_core.language_models import BaseChatModel

from ..prompts.confluence_agent import CONFLUENCE_AGENT_PROMPT
from .base import BaseNode

logger = logging.getLogger(__name__)


class ConfluenceAgentNode(BaseNode):
    """Confluence agent that only sees Confluence tools."""

    def __init__(self, llm_with_tools: BaseChatModel):
        """Initialize the Confluence agent node.

        Args:
            llm_with_tools: LLM with Confluence tools bound (only Confluence tools)
        """
        super().__init__("confluence_agent")
        self.llm_with_tools = llm_with_tools

    def __call__(self, state: dict) -> dict:
        """Execute Confluence agent reasoning."""
        self.log_node_start()
        messages = state["messages"]

        prompt = CONFLUENCE_AGENT_PROMPT.invoke({"messages": messages})
        logger.info("Invoking Confluence agent LLM...")
        response = self.llm_with_tools.invoke(prompt.messages)

        # Log tool calls if any
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"Confluence agent requested {len(response.tool_calls)} tool call(s):")
            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                logger.info(f"  - {tool_name}")
        else:
            logger.info("Confluence agent response (no tool calls)")

        self.log_node_end()
        return {"messages": [response]}

