"""Final responder node for generating user-facing responses."""

import logging

from langchain_core.language_models import BaseChatModel

from ..prompts.final_responder import FINAL_RESPONDER_PROMPT
from .base import BaseNode

logger = logging.getLogger(__name__)


class FinalResponderNode(BaseNode):
    """Final responder that generates user-facing responses."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the final responder node.

        Args:
            llm: LLM for generating responses (no tools)
        """
        super().__init__("final_responder")
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        """Generate final response for user."""
        self.log_node_start()
        messages = state["messages"]

        prompt = FINAL_RESPONDER_PROMPT.invoke({"messages": messages})
        logger.info("Invoking final responder LLM...")
        response = self.llm.invoke(prompt.messages)

        self.log_node_end()
        return {"messages": [response]}

