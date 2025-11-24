"""Confluence export nodes."""

import json
import logging
import os
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .base import BaseNode
from .utils import extract_content_text

logger = logging.getLogger(__name__)


class EnsureAnalysisContextNode(BaseNode):
    """Check that we have recent analysis results in state."""

    def __init__(self):
        """Initialize the node."""
        super().__init__("ensure_analysis_context")

    def __call__(self, state: dict) -> dict:
        """Ensure analysis context exists."""
        self.log_node_start()
        messages = state["messages"]

        # Try to extract analysis context from recent messages
        analysis_found = False
        recent_analysis = {
            "question": None,
            "datasets_used": [],
            "code": None,
            "result_preview": None,
            "plot_path": None,
            "summary": None,
        }

        # Look backwards through messages for analysis results
        for msg in reversed(messages):
            if hasattr(msg, "name") and msg.name == "run_analysis":
                # Found a run_analysis tool result
                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, str):
                        # Try to parse JSON if it's a string
                        try:
                            result = json.loads(content)
                            recent_analysis["result_preview"] = result.get(
                                "result_df_preview"
                            )
                            recent_analysis["plot_path"] = result.get("plot_path")
                            analysis_found = True
                        except (ValueError, TypeError, json.JSONDecodeError):
                            pass
                    elif isinstance(content, dict):
                        recent_analysis["result_preview"] = content.get(
                            "result_df_preview"
                        )
                        recent_analysis["plot_path"] = content.get("plot_path")
                        analysis_found = True
                break

        # Look for the original question
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                recent_analysis["question"] = (
                    msg.content if hasattr(msg, "content") else str(msg)
                )
                break

        # Look for code in tool calls
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if getattr(tool_call, "name", "") == "run_analysis":
                        args = getattr(tool_call, "args", {})
                        recent_analysis["code"] = args.get("code")
                        recent_analysis["datasets_used"] = args.get("dataset_ids", [])
                        break
                if recent_analysis["code"]:
                    break

        # Look for summary in assistant messages
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                if hasattr(msg, "content") and msg.content:
                    recent_analysis["summary"] = msg.content
                    break

        if not analysis_found and not recent_analysis.get("code"):
            # No analysis context found
            error_msg = AIMessage(
                content="I don't have any recent analysis results to export to Confluence. Please run an analysis first, then ask me to create a Confluence report."
            )
            logger.warning("No analysis context found for Confluence export")
            self.log_node_end()
            return {"messages": [error_msg]}

        # Store analysis context
        logger.info("Analysis context found and stored")
        self.log_node_end()
        return {"analysis_context": recent_analysis}


class BuildConfluencePageDraftNode(BaseNode):
    """Use LLM to generate a Confluence-ready page draft from analysis context."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the node.

        Args:
            llm: LLM for generating page draft
        """
        super().__init__("build_confluence_page_draft")
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        """Build Confluence page draft."""
        self.log_node_start()
        analysis_context = state.get("analysis_context", {})
        if not analysis_context:
            error_msg = AIMessage(
                content="No analysis context available to build Confluence page."
            )
            return {"messages": [error_msg]}

        question = analysis_context.get("question", "Data Analysis")
        datasets = analysis_context.get("datasets_used", [])
        code = analysis_context.get("code", "")
        result_preview = analysis_context.get("result_preview", "")
        plot_path = analysis_context.get("plot_path", "")
        summary = analysis_context.get("summary", "")

        date_str = datetime.now().strftime("%Y-%m-%d")

        system_prompt = f"""You are creating a Confluence page draft from a data analysis.

Create a well-structured Confluence page in markdown format with the following sections:
1. **Overview / Business Question**: Brief overview and the original question
2. **Datasets Used**: List the datasets that were analyzed
3. **Methodology**: Include the Python code used (in a code block)
4. **Results**: Describe the results, include data preview if available, and reference plots if created
5. **Interpretation / Caveats**: Key insights and any limitations or assumptions
6. **Reproduction Steps**: Brief steps to reproduce the analysis

Format the page title as: "[Agent] {question[:50]} – {date_str}"

Use Confluence markdown syntax. For code blocks, use ```python.
For tables, use markdown table syntax.
For images, use: ![{plot_path}]({plot_path}) if a plot exists.

Be concise but comprehensive."""

        user_prompt = f"""Create a Confluence page draft for this analysis:

**Question**: {question}

**Datasets Used**: {", ".join(datasets) if datasets else "Not specified"}

**Code**:
```python
{code}
```

**Result Preview**:
{result_preview if result_preview else "No preview available"}

**Plot**: {plot_path if plot_path else "No plot generated"}

**Summary**:
{summary if summary else "No summary available"}

Generate the full Confluence page content in markdown format."""

        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(prompt_messages)
        page_content = extract_content_text(getattr(response, "content", None))

        # Extract title (first line or generate from question)
        lines = page_content.split("\n")
        title = (
            lines[0].replace("#", "").strip()
            if lines
            else f"[Agent] {question[:50]} – {date_str}"
        )
        if title.startswith("[Agent]"):
            body_markdown = "\n".join(lines[1:]).strip()
        else:
            title = f"[Agent] {question[:50]} – {date_str}"
            body_markdown = page_content.strip()

        draft = {
            "title": title,
            "body_markdown": body_markdown,
        }

        logger.info(f"Generated Confluence page draft with title: {title[:50]}...")
        self.log_node_end()

        # Store draft in analysis_context
        analysis_context["confluence_draft"] = draft
        return {"analysis_context": analysis_context}


class DecideConfluenceDestinationNode(BaseNode):
    """Decide where to save the Confluence page (space, parent page)."""

    def __init__(self, confluence_space_key_analytics: str):
        """Initialize the node.

        Args:
            confluence_space_key_analytics: Default Confluence space key
        """
        super().__init__("decide_confluence_destination")
        self.confluence_space_key_analytics = confluence_space_key_analytics

    def __call__(self, state: dict) -> dict:
        """Decide Confluence destination."""
        self.log_node_start()
        # For now, use default space from settings
        # In the future, could extract from user message or use LLM
        space_key = self.confluence_space_key_analytics
        parent_page_id = None  # Could be extracted from user message in future

        logger.info(f"Decided destination: space={space_key}, parent={parent_page_id}")
        self.log_node_end()

        analysis_context = state.get("analysis_context", {})
        analysis_context["confluence_destination"] = {
            "space_key": space_key,
            "parent_page_id": parent_page_id,
        }
        return {"analysis_context": analysis_context}


class CreateConfluencePageNode(BaseNode):
    """Call Confluence MCP tool to create the page."""

    def __init__(self, tools: list, confluence_space_key_analytics: str):
        """Initialize the node.

        Args:
            tools: List of available tools
            confluence_space_key_analytics: Default Confluence space key
        """
        super().__init__("create_confluence_page")
        self.tools = tools
        self.confluence_space_key_analytics = confluence_space_key_analytics

    async def __call__(self, state: dict) -> dict:
        """Create Confluence page."""
        self.log_node_start()
        analysis_context = state.get("analysis_context", {})
        draft = analysis_context.get("confluence_draft")
        destination = analysis_context.get("confluence_destination", {})

        if not draft:
            error_msg = AIMessage(
                content="No Confluence page draft available. Cannot create page."
            )
            return {"messages": [error_msg]}

        space_key = destination.get("space_key", self.confluence_space_key_analytics)
        parent_page_id = destination.get("parent_page_id")

        # Find Confluence tools
        confluence_tool = None
        available_tool_names = [tool.name for tool in self.tools]
        logger.info(f"Available tools for create_page: {available_tool_names}")

        for tool in self.tools:
            tool_name = tool.name.lower()
            if (
                "confluence" in tool_name
                and "create" in tool_name
                and "page" in tool_name
            ):
                confluence_tool = tool
                logger.info(f"Found Confluence create tool: {tool.name}")
                break

        if not confluence_tool:
            # Try alternative names
            for tool in self.tools:
                tool_name = tool.name.lower()
                if "create" in tool_name and "page" in tool_name:
                    confluence_tool = tool
                    logger.info(f"Found create tool (alternative): {tool.name}")
                    break

        if not confluence_tool:
            error_msg = AIMessage(
                content=f"Confluence MCP tools are not available. Please ensure CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN are configured in your .env file to enable Confluence tools in the unified MCP server.\n\n"
                f"Available tools: {', '.join(available_tool_names)}"
            )
            logger.error(
                f"Confluence create page tool not found. Available tools: {available_tool_names}"
            )
            return {"messages": [error_msg]}

        # Prepare tool arguments
        tool_args = {
            "space_key": space_key,
            "title": draft["title"],
            "body": draft["body_markdown"],
        }
        if parent_page_id:
            tool_args["parent_id"] = parent_page_id

        try:
            # Call the tool (async)
            result = await confluence_tool.ainvoke(tool_args)

            # Parse result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass

            # Check for error in result
            if isinstance(result, dict) and "error" in result:
                error_msg_text = result.get("error", "Unknown error")
                logger.error(f"Confluence tool returned error: {error_msg_text}")
                logger.error(f"Full result: {result}")

                # Provide helpful guidance for common errors
                if (
                    "permission" in error_msg_text.lower()
                    or "does not have permission" in error_msg_text.lower()
                ):
                    error_msg = AIMessage(
                        content=f"❌ Failed to create Confluence page: {error_msg_text}\n\n"
                        f"**Troubleshooting:**\n"
                        f"1. Check that the Confluence user ({os.getenv('CONFLUENCE_USERNAME', 'configured user')}) has permission to create pages in the '{space_key}' space\n"
                        f"2. Verify the space key '{space_key}' exists and is accessible\n"
                        f"3. Ensure the Confluence API token has the necessary permissions\n"
                        f"4. Try using a different space key if you have access to other spaces"
                    )
                else:
                    error_msg = AIMessage(
                        content=f"❌ Failed to create Confluence page: {error_msg_text}"
                    )
                return {"messages": [error_msg]}

            page_id = result.get("page_id") or result.get("id") or result.get("pageId")
            page_url = (
                result.get("url") or result.get("page_url") or result.get("pageUrl")
            )

            logger.info(f"Confluence page created: id={page_id}, url={page_url}")
            logger.info(f"Full result: {result}")
            self.log_node_end()

            if not page_id:
                logger.error(f"No page_id in result. Full result: {result}")
                error_msg = AIMessage(
                    content=f"Failed to create Confluence page: The page was created but no ID was returned. Result: {result}"
                )
                return {"messages": [error_msg]}

            return {
                "confluence_page_id": page_id,
                "confluence_page_url": page_url,
            }
        except Exception as e:
            logger.error(f"Error creating Confluence page: {e}", exc_info=True)
            error_msg = AIMessage(content=f"Failed to create Confluence page: {str(e)}")
            return {"messages": [error_msg]}


class StoreConfluencePageInfoNode(BaseNode):
    """Store Confluence page info and return user-friendly response."""

    def __init__(self):
        """Initialize the node."""
        super().__init__("store_confluence_page_info")

    def __call__(self, state: dict) -> dict:
        """Store Confluence page info."""
        self.log_node_start()
        page_id = state.get("confluence_page_id")
        page_url = state.get("confluence_page_url")

        if not page_id or not page_url:
            error_msg = AIMessage(
                content="Confluence page was not created successfully."
            )
            return {"messages": [error_msg]}

        success_msg = AIMessage(
            content=f"✅ Successfully created Confluence page!\n\n"
            f"**Page ID**: {page_id}\n"
            f"**URL**: {page_url}\n\n"
            f"You can view and edit the page at the URL above."
        )

        logger.info(f"Stored Confluence page info: id={page_id}, url={page_url}")
        self.log_node_end()

        return {"messages": [success_msg]}
