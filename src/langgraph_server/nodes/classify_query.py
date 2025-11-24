"""Classify query node."""

import json
import logging

from langchain_core.language_models import BaseChatModel

from ..prompts import COMBINED_CLASSIFICATION_PROMPT
from .base import BaseNode

logger = logging.getLogger(__name__)


def extract_content_text(content):
    """Extract text from content, handling both string and list formats."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
                elif "text" in item:
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    return str(content)


class ClassifyQueryNode(BaseNode):
    """Classify the user query into DOCUMENT_QA, DATA_ANALYSIS, or BOTH, and detect doc_action."""

    def __init__(self, llm_json: BaseChatModel):
        """Initialize the classify query node.

        Args:
            llm_json: Low-temperature LLM for structured output
        """
        super().__init__("classify_query")
        self.llm_json = llm_json

    def __call__(self, state: dict) -> dict:
        """Classify the query and return classification."""
        self.log_node_start()
        messages = state["messages"]

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg
                break

        if not user_message:
            # Default to DATA_ANALYSIS if no user message found
            classification = "DATA_ANALYSIS"
            doc_action = "NONE"
        else:
            # Use LLM to classify both query type and doc_action in a single call
            prompt = COMBINED_CLASSIFICATION_PROMPT.invoke({"messages": [user_message]})
            response = self.llm_json.invoke(prompt.messages)
            response_text = extract_content_text(
                getattr(response, "content", None)
            ).strip()

            # Parse JSON response
            try:
                classification_data = json.loads(response_text)
                classification = classification_data.get(
                    "query_classification", "DATA_ANALYSIS"
                )
                doc_action = classification_data.get("doc_action", "NONE")

                # Validate classification values
                if classification not in ["DOCUMENT_QA", "DATA_ANALYSIS", "BOTH"]:
                    logger.warning(
                        f"Invalid query_classification: {classification}, defaulting to DATA_ANALYSIS"
                    )
                    classification = "DATA_ANALYSIS"

                if doc_action not in ["FROM_ANALYSIS", "FROM_CONFLUENCE", "NONE"]:
                    logger.warning(
                        f"Invalid doc_action: {doc_action}, defaulting to NONE"
                    )
                    doc_action = "NONE"

            except (ValueError, json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse JSON response: {e}. Response: {response_text[:200]}"
                )
                # Fallback: try to parse as text (backward compatibility)
                response_upper = response_text.upper()

                # Parse classification
                if "DOCUMENT_QA" in response_upper:
                    classification = "DOCUMENT_QA"
                elif "BOTH" in response_upper:
                    classification = "BOTH"
                else:
                    classification = "DATA_ANALYSIS"

                # Parse doc_action
                if "FROM_ANALYSIS" in response_upper:
                    doc_action = "FROM_ANALYSIS"
                elif "FROM_CONFLUENCE" in response_upper:
                    doc_action = "FROM_CONFLUENCE"
                else:
                    doc_action = "NONE"

        logger.info(f"Query classified as: {classification}")
        logger.info(f"Doc action classified as: {doc_action}")
        self.log_node_end()
        return {
            "query_classification": classification,
            "doc_action": doc_action,
        }
