"""Confluence read nodes."""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..prompts import CONFLUENCE_QUERY_UNDERSTANDING_PROMPT
from .base import BaseNode
from .utils import extract_content_text

logger = logging.getLogger(__name__)


class UnderstandConfluenceQueryNode(BaseNode):
    """Understand and reformulate the user's Confluence query."""

    def __init__(self, llm_json: BaseChatModel):
        """Initialize the node.

        Args:
            llm_json: Low-temperature LLM for structured output
        """
        super().__init__("understand_confluence_query")
        self.llm_json = llm_json

    def __call__(self, state: dict) -> dict:
        """Understand Confluence query."""
        self.log_node_start()
        messages = state["messages"]

        # Get the user's query
        user_query = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content if hasattr(msg, "content") else str(msg)
                break

        if not user_query:
            error_msg = AIMessage(content="No query provided for Confluence search.")
            return {"messages": [error_msg]}

        # Use LLM to understand and reformulate the query
        prompt = CONFLUENCE_QUERY_UNDERSTANDING_PROMPT.invoke(
            {"messages": [HumanMessage(content=user_query)]}
        )
        response = self.llm_json.invoke(prompt.messages)
        response_text = extract_content_text(getattr(response, "content", None)).strip()

        # Parse JSON response
        try:
            query_info = json.loads(response_text)
        except (ValueError, json.JSONDecodeError):
            logger.warning(
                f"Could not parse LLM response as JSON: {response_text[:200]}. Using fallback."
            )
            # Fallback: create a default reformulation
            query_info = {
                "query_type": "SPECIFIC_SEARCH",
                "search_query": user_query,
                "is_meta_question": False,
                "explanation": "Using original query as fallback after parse failure",
            }

        search_query = query_info.get("search_query", user_query)
        query_type = query_info.get("query_type", "SPECIFIC_SEARCH")
        is_meta_question = query_info.get("is_meta_question", False)

        logger.info(f"Query type: {query_type}")
        logger.info(f"Reformulated search query: {search_query}")
        logger.info(f"Is meta-question: {is_meta_question}")
        self.log_node_end()

        # Store query understanding in state
        analysis_context = state.get("analysis_context", {})
        analysis_context["confluence_query_info"] = {
            "original_query": user_query,
            "search_query": search_query,
            "query_type": query_type,
            "is_meta_question": is_meta_question,
        }
        return {"analysis_context": analysis_context}


class ConfluenceSearchNode(BaseNode):
    """Search Confluence pages based on reformulated query."""

    def __init__(self, tools: list):
        """Initialize the node.

        Args:
            tools: List of available tools
        """
        super().__init__("confluence_search")
        self.tools = tools

    async def __call__(self, state: dict) -> dict:
        """Search Confluence."""
        self.log_node_start()
        analysis_context = state.get("analysis_context", {})
        query_info = analysis_context.get("confluence_query_info", {})

        if not query_info:
            error_msg = AIMessage(
                content="Query understanding failed. Cannot search Confluence."
            )
            return {"messages": [error_msg]}

        search_query = query_info.get("search_query", "")
        is_meta_question = query_info.get("is_meta_question", False)

        # Find Confluence search tool
        confluence_tool = None
        available_tool_names = [tool.name for tool in self.tools]
        logger.info(f"Available tools: {available_tool_names}")

        for tool in self.tools:
            tool_name = tool.name.lower()
            logger.debug(f"Checking tool: {tool.name} (lowercase: {tool_name})")
            if "confluence" in tool_name and "search" in tool_name:
                confluence_tool = tool
                logger.info(f"Found Confluence search tool: {tool.name}")
                break

        if not confluence_tool:
            # Try alternative names
            for tool in self.tools:
                tool_name = tool.name.lower()
                if "search" in tool_name and "page" in tool_name:
                    confluence_tool = tool
                    logger.info(f"Found search tool (alternative): {tool.name}")
                    break

        if not confluence_tool:
            error_msg = AIMessage(
                content=f"Confluence MCP tools are not available. Please ensure CONFLUENCE_MCP_SERVER_URL is configured and the Confluence MCP server is running.\n\n"
                f"Available tools: {', '.join(available_tool_names)}"
            )
            logger.error(
                f"Confluence search tool not found. Available tools: {available_tool_names}"
            )
            return {"messages": [error_msg]}

        # Prepare tool arguments
        tool_args = {
            "query": search_query,
            "limit": 20
            if is_meta_question
            else 10,  # Get more results for meta-questions
        }

        try:
            # Call the tool (async)
            result = await confluence_tool.ainvoke(tool_args)

            # Parse result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Extract pages list
            pages = (
                result.get("pages")
                or result.get("results")
                or result.get("items")
                or []
            )

            logger.info(f"Found {len(pages)} Confluence pages")
            self.log_node_end()

            # Store search results in state
            analysis_context["confluence_search_results"] = pages
            analysis_context["confluence_is_meta_question"] = is_meta_question
            return {"analysis_context": analysis_context}
        except Exception as e:
            logger.error(f"Error searching Confluence: {e}")
            error_msg = AIMessage(content=f"Failed to search Confluence: {str(e)}")
            return {"messages": [error_msg]}


class SelectConfluencePageNode(BaseNode):
    """Select the most relevant Confluence page(s) from search results, or handle meta-questions."""

    def __init__(self, llm_json: BaseChatModel):
        """Initialize the node.

        Args:
            llm_json: Low-temperature LLM for structured output
        """
        super().__init__("select_confluence_page")
        self.llm_json = llm_json

    def __call__(self, state: dict) -> dict:
        """Select Confluence page."""
        self.log_node_start()
        analysis_context = state.get("analysis_context", {})
        search_results = analysis_context.get("confluence_search_results", [])
        is_meta_question = analysis_context.get("confluence_is_meta_question", False)
        query_info = analysis_context.get("confluence_query_info", {})
        original_query = query_info.get("original_query", "")

        if not search_results:
            error_msg = AIMessage(
                content="No Confluence pages found matching your query. Please try a different search term."
            )
            return {"messages": [error_msg]}

        # For meta-questions, we should show a sample of pages rather than selecting one
        if is_meta_question:
            # Format the pages for display
            pages_summary = "Here are some Confluence pages I found:\n\n"
            for i, page in enumerate(search_results[:10], 1):  # Show up to 10 pages
                title = page.get("title") or page.get("name") or "Untitled"
                url = (
                    page.get("url") or page.get("page_url") or page.get("pageUrl") or ""
                )
                space_key = page.get("space_key", "")

                pages_summary += f"{i}. **{title}**"
                if space_key:
                    pages_summary += f" (Space: {space_key})"
                if url:
                    pages_summary += f"\n   URL: {url}"
                pages_summary += "\n\n"

            response_msg = AIMessage(
                content=f"I found {len(search_results)} Confluence page(s). Here's a sample:\n\n{pages_summary}\n"
                f"To get more details about a specific page, you can ask me to summarize it or ask a specific question about it."
            )
            logger.info(f"Handled meta-question: showing {len(search_results)} pages")
            self.log_node_end()
            return {"messages": [response_msg]}

        # For specific searches, select the most relevant page
        messages = state["messages"]
        user_query = original_query
        if not user_query:
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    user_query = msg.content if hasattr(msg, "content") else str(msg)
                    break

        # Use LLM to select the best page
        # Format search results for LLM
        results_text = "Search results:\n"
        for i, page in enumerate(search_results[:5]):  # Limit to top 5
            page_id = page.get("id") or page.get("page_id") or page.get("pageId")
            title = page.get("title") or page.get("name") or "Untitled"
            excerpt = (
                page.get("excerpt") or page.get("snippet") or page.get("body") or ""
            )
            url = page.get("url") or page.get("page_url") or page.get("pageUrl") or ""

            results_text += f"\n{i + 1}. Title: {title}\n"
            if excerpt:
                results_text += f"   Excerpt: {excerpt[:200]}...\n"
            if url:
                results_text += f"   URL: {url}\n"
            results_text += f"   ID: {page_id}\n"

        system_prompt = """You are a function-calling engine inside a deterministic agent.

You MUST output ONLY one word or number and NOTHING else.

Rules:
- No natural language before or after the selection.
- No backticks, no comments, no explanations.
- Output ONLY the number (1-5) or "NONE".

Given the user's query and the search results, select the ONE most relevant page.
Respond with ONLY the number (1-5) of the most relevant page.
If none are relevant, respond with "NONE"."""

        user_prompt = f"""User query: {user_query}

{results_text}

Which page is most relevant? Respond with the number (1-5) or "NONE"."""

        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Use low-temperature LLM for structured output
        response = self.llm_json.invoke(prompt_messages)
        selection_text = response.content.strip()

        # Parse selection
        try:
            if "NONE" in selection_text.upper():
                error_msg = AIMessage(
                    content="I couldn't find a relevant Confluence page matching your query. Please try a different search term."
                )
                return {"messages": [error_msg]}

            # Extract number
            numbers = re.findall(r"\d+", selection_text)
            if numbers:
                selected_idx = int(numbers[0]) - 1
                if 0 <= selected_idx < len(search_results):
                    selected_page = search_results[selected_idx]
                    logger.info(
                        f"Selected page: {selected_page.get('title', 'Unknown')}"
                    )
                    self.log_node_end()

                    analysis_context["selected_confluence_page"] = selected_page
                    return {"analysis_context": analysis_context}
        except Exception as e:
            logger.error(f"Error parsing page selection: {e}")

        # Default to first result if parsing fails
        selected_page = search_results[0]
        logger.info(f"Defaulted to first page: {selected_page.get('title', 'Unknown')}")
        self.log_node_end()

        analysis_context["selected_confluence_page"] = selected_page
        return {"analysis_context": analysis_context}


class GetConfluencePageNode(BaseNode):
    """Fetch the full content of the selected Confluence page."""

    def __init__(self, tools: list):
        """Initialize the node.

        Args:
            tools: List of available tools
        """
        super().__init__("get_confluence_page")
        self.tools = tools

    async def __call__(self, state: dict) -> dict:
        """Get Confluence page."""
        self.log_node_start()
        analysis_context = state.get("analysis_context", {})
        selected_page = analysis_context.get("selected_confluence_page")

        if not selected_page:
            error_msg = AIMessage(
                content="No Confluence page selected. Cannot fetch content."
            )
            return {"messages": [error_msg]}

        page_id = (
            selected_page.get("id")
            or selected_page.get("page_id")
            or selected_page.get("pageId")
        )

        if not page_id:
            error_msg = AIMessage(
                content="Selected page has no ID. Cannot fetch content."
            )
            return {"messages": [error_msg]}

        # Find Confluence get page tool
        confluence_tool = None
        available_tool_names = [tool.name for tool in self.tools]
        logger.info(f"Available tools for get_page: {available_tool_names}")

        for tool in self.tools:
            tool_name = tool.name.lower()
            if (
                "confluence" in tool_name
                and ("get" in tool_name or "fetch" in tool_name)
                and "page" in tool_name
            ):
                confluence_tool = tool
                logger.info(f"Found Confluence get tool: {tool.name}")
                break

        if not confluence_tool:
            # Try alternative names
            for tool in self.tools:
                tool_name = tool.name.lower()
                if ("get" in tool_name or "fetch" in tool_name) and "page" in tool_name:
                    confluence_tool = tool
                    logger.info(f"Found get tool (alternative): {tool.name}")
                    break

        if not confluence_tool:
            error_msg = AIMessage(
                content=f"Confluence MCP tools are not available. Please ensure CONFLUENCE_MCP_SERVER_URL is configured and the Confluence MCP server is running.\n\n"
                f"Available tools: {', '.join(available_tool_names)}"
            )
            logger.error(
                f"Confluence get page tool not found. Available tools: {available_tool_names}"
            )
            return {"messages": [error_msg]}

        # Prepare tool arguments
        tool_args = {
            "page_id": page_id,
        }

        try:
            # Call the tool (async)
            result = await confluence_tool.ainvoke(tool_args)

            # Parse result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Extract page content
            page_content = (
                result.get("content") or result.get("body") or result.get("text") or ""
            )
            page_title = result.get("title") or selected_page.get("title") or "Unknown"
            page_url = result.get("url") or selected_page.get("url") or ""

            logger.info(f"Fetched Confluence page: {page_title}")
            self.log_node_end()

            # Store page content
            analysis_context["confluence_page_content"] = {
                "title": page_title,
                "content": page_content,
                "url": page_url,
                "page_id": page_id,
            }
            return {"analysis_context": analysis_context}
        except Exception as e:
            logger.error(f"Error fetching Confluence page: {e}")
            error_msg = AIMessage(content=f"Failed to fetch Confluence page: {str(e)}")
            return {"messages": [error_msg]}


class SummarizeFromConfluenceNode(BaseNode):
    """Summarize or answer questions based on Confluence page content."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the node.

        Args:
            llm: LLM for summarization
        """
        super().__init__("summarize_from_confluence")
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        """Summarize or answer from Confluence page."""
        self.log_node_start()
        analysis_context = state.get("analysis_context", {})
        page_content_data = analysis_context.get("confluence_page_content")

        if not page_content_data:
            error_msg = AIMessage(
                content="No Confluence page content available to summarize."
            )
            return {"messages": [error_msg]}

        page_title = page_content_data.get("title", "Unknown")
        page_content = page_content_data.get("content", "")
        page_url = page_content_data.get("url", "")

        messages = state["messages"]
        user_query = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content if hasattr(msg, "content") else str(msg)
                break

        # Use LLM to summarize or answer
        system_prompt = """You are summarizing or answering questions based on Confluence page content.

If the user asked to "summarize" or "what are the takeaways", provide a clear summary of the page.
If the user asked a specific question, answer it using the page content as context.
Be concise but comprehensive. Cite specific details from the page when relevant."""

        user_prompt = f"""User query: {user_query}

Confluence page title: {page_title}
Confluence page URL: {page_url}

Page content:
{page_content[:8000]}  # Limit content length

Based on this Confluence page, {user_query}"""

        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(prompt_messages)

        # Format response with page reference
        response_content = extract_content_text(getattr(response, "content", None))
        formatted_response = (
            f"{response_content}\n\n*Source: [{page_title}]({page_url})*"
        )
        answer_msg = AIMessage(content=formatted_response)

        logger.info("Generated summary/answer from Confluence page")
        self.log_node_end()

        return {"messages": [answer_msg]}
