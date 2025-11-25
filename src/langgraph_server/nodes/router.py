"""Router node for intent classification."""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..prompts.router import ROUTER_PROMPT
from .base import BaseNode
from .utils import extract_content_text

logger = logging.getLogger(__name__)


class RouterNode(BaseNode):
    """Router node that classifies user intent."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the router node.

        Args:
            llm: Low-temperature LLM for classification (no tools)
        """
        super().__init__("router")
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        """Classify user intent."""
        self.log_node_start()
        messages = state["messages"]

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg
                break

        if not user_message:
            # Default to ANALYSIS if no user message found
            intent = "ANALYSIS"
            logger.warning("No user message found, defaulting to ANALYSIS")
        else:
            # Use LLM to classify intent
            prompt = ROUTER_PROMPT.invoke({"messages": [user_message]})
            response = self.llm.invoke(prompt.messages)
            response_text = extract_content_text(
                getattr(response, "content", None)
            ).strip()

            # Parse JSON response
            try:
                classification_data = json.loads(response_text)
                intent = classification_data.get("intent", "ANALYSIS")

                # Validate intent
                valid_intents = ["ANALYSIS", "KNOWLEDGE", "CONFLUENCE", "OTHER"]
                if intent not in valid_intents:
                    logger.warning(
                        f"Invalid intent: {intent}, defaulting to ANALYSIS"
                    )
                    intent = "ANALYSIS"

            except (ValueError, json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse JSON response: {e}. Response: {response_text[:200]}"
                )
                # Fallback: try to parse as text
                response_upper = response_text.upper()
                if "KNOWLEDGE" in response_upper:
                    intent = "KNOWLEDGE"
                elif "CONFLUENCE" in response_upper:
                    intent = "CONFLUENCE"
                elif "OTHER" in response_upper:
                    intent = "OTHER"
                else:
                    intent = "ANALYSIS"

        logger.info(f"Intent classified as: {intent}")
        self.log_node_end()
        return {
            "intent": intent,
        }

