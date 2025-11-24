"""Document QA node."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from ..prompts import DOCUMENT_QA_PROMPT
from .base import BaseNode

logger = logging.getLogger(__name__)


class DocumentQANode(BaseNode):
    """Handle pure document/terminology questions."""

    def __init__(self, llm_with_tools: BaseChatModel):
        """Initialize the document QA node.

        Args:
            llm_with_tools: LLM with tools bound
        """
        super().__init__("document_qa")
        self.llm_with_tools = llm_with_tools

    def __call__(self, state: dict) -> dict:
        """Handle document QA query."""
        self.log_node_start()
        messages = state["messages"]

        # Add knowledge context if available
        prompt_messages = list(messages)
        if state.get("knowledge_context"):
            prompt_messages.insert(
                0,
                SystemMessage(
                    content=f"Knowledge context:\n{state['knowledge_context']}"
                ),
            )

        prompt = DOCUMENT_QA_PROMPT.invoke({"messages": prompt_messages})
        response = self.llm_with_tools.invoke(prompt.messages)

        logger.info("Document QA response generated")
        self.log_node_end()
        return {"messages": [response]}
