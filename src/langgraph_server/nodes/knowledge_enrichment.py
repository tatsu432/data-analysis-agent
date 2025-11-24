"""Knowledge enrichment node."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage

from ..prompts import KNOWLEDGE_ENRICHMENT_PROMPT
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


class KnowledgeEnrichmentNode(BaseNode):
    """Enrich data analysis queries with document knowledge."""

    def __init__(self, llm: BaseChatModel, tools: list, tool_choice: str | None = None):
        """Initialize the knowledge enrichment node.

        Args:
            llm: Base LLM for knowledge enrichment
            tools: List of available tools
            tool_choice: Tool choice setting for LLM
        """
        super().__init__("knowledge_enrichment")
        self.llm = llm
        self.tools = tools
        self.tool_choice = tool_choice

    def __call__(self, state: dict) -> dict:
        """Enrich query with knowledge context."""
        self.log_node_start()
        messages = state["messages"]

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg
                break

        if not user_message:
            return {"knowledge_context": ""}

        # Quick heuristic: check if query contains common domain terms
        query_text = (
            user_message.content
            if hasattr(user_message, "content")
            else str(user_message)
        )
        query_lower = query_text.lower()

        # Common domain terms that might need lookup
        domain_terms = [
            "gp",
            "hp",
            "at-risk",
            "at risk",
            "trx",
            "rx",
            "ddi",
            "mr activity",
            "unmet medical needs",
            "開発",
            "アンメット",
        ]

        has_domain_terms = any(term in query_lower for term in domain_terms)

        # If no obvious domain terms, skip enrichment to save time
        if not has_domain_terms:
            logger.info(
                "No obvious domain terms detected, skipping knowledge enrichment"
            )
            return {"knowledge_context": ""}

        # Use LLM to identify terms and look them up
        prompt = KNOWLEDGE_ENRICHMENT_PROMPT.invoke({"messages": [user_message]})
        # Use same tool_choice as main LLM for consistency
        if self.tool_choice:
            llm_with_knowledge_tools = self.llm.bind_tools(
                self.tools, tool_choice=self.tool_choice
            )
        else:
            llm_with_knowledge_tools = self.llm.bind_tools(self.tools)
        response = llm_with_knowledge_tools.invoke(prompt.messages)

        # If LLM wants to call tools, we need to execute them
        knowledge_context = ""

        # Check if response has tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Limit to max 3 tool calls to avoid excessive lookups
            tool_calls_to_execute = response.tool_calls[:3]

            # Execute knowledge tool calls
            tool_results = []
            for tool_call in tool_calls_to_execute:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                # Only execute knowledge tools
                if tool_name not in ["get_term_definition", "search_knowledge"]:
                    continue

                # Find the tool
                tool_obj = None
                for t in self.tools:
                    if t.name == tool_name:
                        tool_obj = t
                        break

                if tool_obj:
                    try:
                        result = tool_obj.invoke(tool_args)
                        tool_results.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call.get("id", ""),
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")

            # Only continue if we got tool results
            if tool_results:
                # Get enriched response
                enriched_messages = [user_message, response] + tool_results
                enriched_prompt = KNOWLEDGE_ENRICHMENT_PROMPT.invoke(
                    {"messages": enriched_messages}
                )
                enriched_response = self.llm.invoke(enriched_prompt.messages)

                # Extract knowledge context from response
                knowledge_context = extract_content_text(
                    getattr(enriched_response, "content", None)
                ).strip()
        else:
            # No tool calls, skip enrichment
            knowledge_context = ""

        if knowledge_context:
            logger.info(
                f"Knowledge context generated: {len(knowledge_context)} characters"
            )
        else:
            logger.info("No knowledge context generated, proceeding without enrichment")
        self.log_node_end()
        return {"knowledge_context": knowledge_context}
