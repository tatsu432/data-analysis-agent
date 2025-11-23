"""LangGraph definition for the data analysis agent."""

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import NotRequired, TypedDict

from .llm_utils import initialize_llm
from .mcp_tool_loader import MCPToolLoader
from .prompts import (
    ANALYSIS_PROMPT,
    COMBINED_CLASSIFICATION_PROMPT,
    CONFLUENCE_QUERY_UNDERSTANDING_PROMPT,
    DOCUMENT_QA_PROMPT,
    KNOWLEDGE_ENRICHMENT_PROMPT,
)
from .settings import LLMProvider, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load .env from project root (settings.py also loads it, but load here too for safety)
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()  # Fallback to current directory


class AgentState(TypedDict):
    """State for the data analysis agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: NotRequired[int]
    query_classification: NotRequired[Literal["DOCUMENT_QA", "DATA_ANALYSIS", "BOTH"]]
    knowledge_context: NotRequired[str]
    doc_action: NotRequired[Literal["NONE", "FROM_ANALYSIS", "FROM_CONFLUENCE"]]
    confluence_page_id: NotRequired[str]
    confluence_page_url: NotRequired[str]
    analysis_context: NotRequired[
        dict
    ]  # Stores recent analysis results for Confluence export


async def create_agent():
    """
    Create the data analysis agent graph.

    LLM configuration is loaded from environment variables using the CHAT_NODE prefix.
    For example:
        CHAT_NODE__llm_model_provider=openai
        CHAT_NODE__llm_model_name=gpt-5.1
        CHAT_NODE__temperature=0.1
        CHAT_NODE__api_key=sk-...

    Returns:
        Compiled LangGraph StateGraph
    """
    settings = get_settings()
    logger.info(
        f"Creating agent with model: {settings.chat_llm.llm_model_name} "
        f"(provider: {settings.chat_llm.llm_model_provider})"
    )

    # Load MCP tools
    mcp_tool_loader = MCPToolLoader()
    # Load tools once and keep them for the agent lifetime
    tools = await mcp_tool_loader._load_all_servers()

    # Log loaded tools for debugging
    tool_names = [tool.name for tool in tools]
    logger.info("=" * 60)
    logger.info("Agent initialized with %d tools:", len(tools))
    for tool_name in tool_names:
        logger.info("  - %s", tool_name)
    logger.info("=" * 60)

    # Initialize the LLM from settings
    llm = initialize_llm(settings.chat_llm)

    # Create a low-temperature LLM for JSON nodes (0.0-0.2)
    json_temperature = min(0.2, max(0.0, settings.chat_llm.temperature))

    # For JSON nodes, also enable JSON mode if using QwenOllama
    json_llm_config = None
    if (
        json_temperature != settings.chat_llm.temperature
        or settings.chat_llm.llm_model_provider == LLMProvider.QWEN_OLLAMA
    ):
        # Use Pydantic's model_copy to create a copy with modified temperature
        json_llm_config = settings.chat_llm.model_copy(
            update={"temperature": json_temperature}
        )

        # Enable JSON mode for QwenOllama by passing response_format through llm_params
        if settings.chat_llm.llm_model_provider == LLMProvider.QWEN_OLLAMA:
            logger.info("Enabled JSON mode for QwenOllama in JSON nodes")
            # Pass response_format through llm_params so initialize_llm can handle it
            json_llm_config = json_llm_config.model_copy(
                update={
                    "llm_params": {
                        **json_llm_config.llm_params,
                        "response_format": {"type": "json_object"},
                    },
                }
            )
        llm_json = initialize_llm(json_llm_config)
    else:
        llm_json = llm

    # Bind tools to the LLM
    # For Qwen models, use tool_choice for stable behavior
    tool_choice = None
    if settings.chat_llm.llm_model_provider == LLMProvider.QWEN_OLLAMA:
        tool_choice = getattr(settings.chat_llm, "tool_choice", "required")
        logger.info(f"Using tool_choice='{tool_choice}' for QwenOllama")

    if tool_choice:
        llm_with_tools = llm.bind_tools(tools, tool_choice=tool_choice)
    else:
        llm_with_tools = llm.bind_tools(tools)

    # Create classifier node
    def classify_query(state: AgentState):
        """Classify the user query into DOCUMENT_QA, DATA_ANALYSIS, or BOTH, and detect doc_action."""
        messages = state["messages"]
        logger.info("=" * 60)
        logger.info("NODE: classify_query")

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
            # Use LLM to classify both query type and doc_action in a single call (use low-temperature LLM for structured output)
            prompt = COMBINED_CLASSIFICATION_PROMPT.invoke({"messages": [user_message]})
            response = llm_json.invoke(prompt.messages)
            response_text = response.content.strip()

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
        logger.info("=" * 60)
        return {
            "query_classification": classification,
            "doc_action": doc_action,
        }

    # Create document QA node
    def document_qa_node(state: AgentState):
        """Handle pure document/terminology questions."""
        messages = state["messages"]
        logger.info("=" * 60)
        logger.info("NODE: document_qa")

        # Add knowledge context if available
        prompt_messages = list(messages)
        if state.get("knowledge_context"):
            from langchain_core.messages import SystemMessage

            prompt_messages.insert(
                0,
                SystemMessage(
                    content=f"Knowledge context:\n{state['knowledge_context']}"
                ),
            )

        prompt = DOCUMENT_QA_PROMPT.invoke({"messages": prompt_messages})
        response = llm_with_tools.invoke(prompt.messages)

        logger.info("Document QA response generated")
        return {"messages": [response]}

    # Create knowledge enrichment node
    def knowledge_enrichment_node(state: AgentState):
        """Enrich data analysis queries with document knowledge."""
        messages = state["messages"]
        logger.info("=" * 60)
        logger.info("NODE: knowledge_enrichment")

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
        if tool_choice:
            llm_with_knowledge_tools = llm.bind_tools(tools, tool_choice=tool_choice)
        else:
            llm_with_knowledge_tools = llm.bind_tools(tools)
        response = llm_with_knowledge_tools.invoke(prompt.messages)

        # If LLM wants to call tools, we need to execute them
        knowledge_context = ""

        # Check if response has tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Limit to max 3 tool calls to avoid excessive lookups
            tool_calls_to_execute = response.tool_calls[:3]

            # Execute knowledge tool calls
            from langchain_core.messages import ToolMessage

            tool_results = []
            for tool_call in tool_calls_to_execute:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                # Only execute knowledge tools
                if tool_name not in ["get_term_definition", "search_knowledge"]:
                    continue

                # Find the tool
                tool_obj = None
                for t in tools:
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
                enriched_response = llm.invoke(enriched_prompt.messages)

                # Extract knowledge context from response
                knowledge_context = enriched_response.content.strip()
        else:
            # No tool calls, skip enrichment
            knowledge_context = ""

        if knowledge_context:
            logger.info(
                f"Knowledge context generated: {len(knowledge_context)} characters"
            )
        else:
            logger.info("No knowledge context generated, proceeding without enrichment")
        logger.info("=" * 60)
        return {"knowledge_context": knowledge_context}

    # Create call_model function with access to llm_with_tools
    def call_model(state: AgentState):
        """Call the LLM with the current state."""
        messages = state["messages"]
        logger.info("=" * 60)
        logger.info("NODE: agent (call_model)")
        logger.info(f"Number of messages in state: {len(messages)}")
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content") and last_message.content:
                logger.debug(
                    f"Last message content: {str(last_message.content)[:200]}..."
                )
        # Add knowledge context if available and non-empty
        prompt_messages = list(messages)
        knowledge_context = state.get("knowledge_context", "")
        if (
            knowledge_context
            and knowledge_context.strip()
            and len(knowledge_context.strip()) > 10
        ):
            from langchain_core.messages import SystemMessage

            # Insert knowledge context at the beginning
            knowledge_msg = SystemMessage(
                content=f"Document knowledge context:\n{knowledge_context}\n\nUse this context ONLY to understand domain terms and map them to dataset columns. Do NOT let this distract from your primary task of data analysis."
            )
            prompt_messages.insert(0, knowledge_msg)

        # Check if we have tool responses but haven't called run_analysis yet
        # This helps prevent Qwen (and other models) from stopping after list_datasets
        from langchain_core.messages import ToolMessage

        has_tool_responses = False
        has_run_analysis = False
        for msg in messages:
            # Check for tool responses (ToolMessage)
            if isinstance(msg, ToolMessage):
                has_tool_responses = True
                # Check if run_analysis was called
                tool_name = getattr(msg, "name", "")
                if tool_name == "run_analysis":
                    has_run_analysis = True
            # Also check for tool calls in AIMessage
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = getattr(tool_call, "name", "")
                    if tool_name == "run_analysis":
                        has_run_analysis = True

        # If we have tool responses but no run_analysis yet, inject a reminder
        # This is especially important for Qwen which might stop after list_datasets
        if has_tool_responses and not has_run_analysis:
            from langchain_core.messages import SystemMessage

            reminder_msg = SystemMessage(
                content="REMINDER: You have received tool responses (e.g., from list_datasets or get_dataset_schema). "
                "These are INTERMEDIATE INFORMATION, not final answers. "
                "You MUST continue the workflow by generating Python code and calling run_analysis to execute the actual data analysis. "
                "DO NOT stop and present tool responses as your final answer. "
                "The workflow is only complete after you have called run_analysis and received actual analysis results."
            )
            prompt_messages.insert(0, reminder_msg)
            logger.info(
                "Injected reminder: tool responses detected but run_analysis not called yet"
            )

        prompt = ANALYSIS_PROMPT.invoke({"messages": prompt_messages})
        logger.info("Invoking LLM...")
        response = llm_with_tools.invoke(prompt.messages)

        # Log tool calls if any
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"LLM requested {len(response.tool_calls)} tool call(s):")
            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                logger.info(f"  - {tool_name}")
        else:
            logger.info("LLM response (no tool calls)")

        return {"messages": [response]}

    # Confluence Export Subflow Nodes
    settings = get_settings()

    def ensure_analysis_context(state: AgentState):
        """Check that we have recent analysis results in state."""
        logger.info("=" * 60)
        logger.info("NODE: ensure_analysis_context")

        messages = state["messages"]

        # Try to extract analysis context from recent messages
        # Look for tool messages with run_analysis results
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
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content="I don't have any recent analysis results to export to Confluence. Please run an analysis first, then ask me to create a Confluence report."
            )
            logger.warning("No analysis context found for Confluence export")
            logger.info("=" * 60)
            return {"messages": [error_msg]}

        # Store analysis context
        logger.info("Analysis context found and stored")
        logger.info("=" * 60)
        return {"analysis_context": recent_analysis}

    def build_confluence_page_draft(state: AgentState):
        """Use LLM to generate a Confluence-ready page draft from analysis context."""
        logger.info("=" * 60)
        logger.info("NODE: build_confluence_page_draft")

        analysis_context = state.get("analysis_context", {})
        if not analysis_context:
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content="No analysis context available to build Confluence page."
            )
            return {"messages": [error_msg]}

        # Build prompt for LLM to create Confluence page draft
        from datetime import datetime

        from langchain_core.messages import HumanMessage, SystemMessage

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

        response = llm.invoke(prompt_messages)
        page_content = response.content

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
        logger.info("=" * 60)

        # Store draft in analysis_context
        analysis_context["confluence_draft"] = draft
        return {"analysis_context": analysis_context}

    def decide_confluence_destination(state: AgentState):
        """Decide where to save the Confluence page (space, parent page)."""
        logger.info("=" * 60)
        logger.info("NODE: decide_confluence_destination")

        # For now, use default space from settings
        # In the future, could extract from user message or use LLM
        space_key = settings.confluence_space_key_analytics
        parent_page_id = None  # Could be extracted from user message in future

        logger.info(f"Decided destination: space={space_key}, parent={parent_page_id}")
        logger.info("=" * 60)

        analysis_context = state.get("analysis_context", {})
        analysis_context["confluence_destination"] = {
            "space_key": space_key,
            "parent_page_id": parent_page_id,
        }
        return {"analysis_context": analysis_context}

    async def create_confluence_page(state: AgentState):
        """Call Confluence MCP tool to create the page."""
        logger.info("=" * 60)
        logger.info("NODE: create_confluence_page")

        analysis_context = state.get("analysis_context", {})
        draft = analysis_context.get("confluence_draft")
        destination = analysis_context.get("confluence_destination", {})

        if not draft:
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content="No Confluence page draft available. Cannot create page."
            )
            return {"messages": [error_msg]}

        space_key = destination.get(
            "space_key", settings.confluence_space_key_analytics
        )
        parent_page_id = destination.get("parent_page_id")

        # Find Confluence tools - they should be available from the MCP server
        # Common tool names: confluence_create_page, confluence.create_page, create_page
        confluence_tool = None

        # Debug: Log all available tool names
        available_tool_names = [tool.name for tool in tools]
        logger.info(f"Available tools for create_page: {available_tool_names}")

        for tool in tools:
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
            for tool in tools:
                tool_name = tool.name.lower()
                if "create" in tool_name and "page" in tool_name:
                    confluence_tool = tool
                    logger.info(f"Found create tool (alternative): {tool.name}")
                    break

        if not confluence_tool:
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content=f"Confluence MCP tools are not available. Please ensure CONFLUENCE_MCP_SERVER_URL is configured and the Confluence MCP server is running.\n\n"
                f"Available tools: {', '.join(available_tool_names)}"
            )
            logger.error(
                f"Confluence create page tool not found. Available tools: {available_tool_names}"
            )
            return {"messages": [error_msg]}

        # Prepare tool arguments - adapt based on actual tool schema
        # Common parameters: space_key, title, body, parent_id
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

            # Parse result - adapt based on actual tool response format
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
                from langchain_core.messages import AIMessage

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
            logger.info("=" * 60)

            if not page_id:
                logger.error(f"No page_id in result. Full result: {result}")
                from langchain_core.messages import AIMessage

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
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(content=f"Failed to create Confluence page: {str(e)}")
            return {"messages": [error_msg]}

    def store_confluence_page_info_and_respond(state: AgentState):
        """Store Confluence page info and return user-friendly response."""
        logger.info("=" * 60)
        logger.info("NODE: store_confluence_page_info_and_respond")

        page_id = state.get("confluence_page_id")
        page_url = state.get("confluence_page_url")

        if not page_id or not page_url:
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content="Confluence page was not created successfully."
            )
            return {"messages": [error_msg]}

        from langchain_core.messages import AIMessage

        success_msg = AIMessage(
            content=f"✅ Successfully created Confluence page!\n\n"
            f"**Page ID**: {page_id}\n"
            f"**URL**: {page_url}\n\n"
            f"You can view and edit the page at the URL above."
        )

        logger.info(f"Stored Confluence page info: id={page_id}, url={page_url}")
        logger.info("=" * 60)

        return {"messages": [success_msg]}

    # Confluence Read Subflow Nodes

    def understand_confluence_query(state: AgentState):
        """Understand and reformulate the user's Confluence query."""
        logger.info("=" * 60)
        logger.info("NODE: understand_confluence_query")

        messages = state["messages"]

        # Get the user's query
        user_query = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content if hasattr(msg, "content") else str(msg)
                break

        if not user_query:
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(content="No query provided for Confluence search.")
            return {"messages": [error_msg]}

        # Use LLM to understand and reformulate the query (use low-temperature LLM for JSON output)
        from langchain_core.messages import HumanMessage

        prompt = CONFLUENCE_QUERY_UNDERSTANDING_PROMPT.invoke(
            {"messages": [HumanMessage(content=user_query)]}
        )
        response = llm_json.invoke(prompt.messages)
        response_text = response.content.strip()

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
        logger.info("=" * 60)

        # Store query understanding in state
        analysis_context = state.get("analysis_context", {})
        analysis_context["confluence_query_info"] = {
            "original_query": user_query,
            "search_query": search_query,
            "query_type": query_type,
            "is_meta_question": is_meta_question,
        }
        return {"analysis_context": analysis_context}

    async def confluence_search_node(state: AgentState):
        """Search Confluence pages based on reformulated query."""
        logger.info("=" * 60)
        logger.info("NODE: confluence_search_node")

        analysis_context = state.get("analysis_context", {})
        query_info = analysis_context.get("confluence_query_info", {})

        if not query_info:
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content="Query understanding failed. Cannot search Confluence."
            )
            return {"messages": [error_msg]}

        search_query = query_info.get("search_query", "")
        is_meta_question = query_info.get("is_meta_question", False)

        # Find Confluence search tool
        confluence_tool = None

        # Debug: Log all available tool names
        available_tool_names = [tool.name for tool in tools]
        logger.info(f"Available tools: {available_tool_names}")

        for tool in tools:
            tool_name = tool.name.lower()
            logger.debug(f"Checking tool: {tool.name} (lowercase: {tool_name})")
            if "confluence" in tool_name and "search" in tool_name:
                confluence_tool = tool
                logger.info(f"Found Confluence search tool: {tool.name}")
                break

        if not confluence_tool:
            # Try alternative names
            for tool in tools:
                tool_name = tool.name.lower()
                if "search" in tool_name and "page" in tool_name:
                    confluence_tool = tool
                    logger.info(f"Found search tool (alternative): {tool.name}")
                    break

        if not confluence_tool:
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content=f"Confluence MCP tools are not available. Please ensure CONFLUENCE_MCP_SERVER_URL is configured and the Confluence MCP server is running.\n\n"
                f"Available tools: {', '.join(available_tool_names)}"
            )
            logger.error(
                f"Confluence search tool not found. Available tools: {available_tool_names}"
            )
            return {"messages": [error_msg]}

        # Prepare tool arguments - adapt based on actual tool schema
        # For meta-questions, use a broader search or increase limit
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
                import json

                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Extract pages list - adapt based on actual tool response format
            pages = (
                result.get("pages")
                or result.get("results")
                or result.get("items")
                or []
            )

            logger.info(f"Found {len(pages)} Confluence pages")
            logger.info("=" * 60)

            # Store search results in state
            analysis_context["confluence_search_results"] = pages
            analysis_context["confluence_is_meta_question"] = is_meta_question
            return {"analysis_context": analysis_context}
        except Exception as e:
            logger.error(f"Error searching Confluence: {e}")
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(content=f"Failed to search Confluence: {str(e)}")
            return {"messages": [error_msg]}

    def select_confluence_page(state: AgentState):
        """Select the most relevant Confluence page(s) from search results, or handle meta-questions."""
        logger.info("=" * 60)
        logger.info("NODE: select_confluence_page")

        analysis_context = state.get("analysis_context", {})
        search_results = analysis_context.get("confluence_search_results", [])
        is_meta_question = analysis_context.get("confluence_is_meta_question", False)
        query_info = analysis_context.get("confluence_query_info", {})
        original_query = query_info.get("original_query", "")

        if not search_results:
            from langchain_core.messages import AIMessage

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

            from langchain_core.messages import AIMessage

            response_msg = AIMessage(
                content=f"I found {len(search_results)} Confluence page(s). Here's a sample:\n\n{pages_summary}\n"
                f"To get more details about a specific page, you can ask me to summarize it or ask a specific question about it."
            )
            logger.info(f"Handled meta-question: showing {len(search_results)} pages")
            logger.info("=" * 60)
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
        from langchain_core.messages import HumanMessage, SystemMessage

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
        response = llm_json.invoke(prompt_messages)
        selection_text = response.content.strip()

        # Parse selection
        try:
            if "NONE" in selection_text.upper():
                from langchain_core.messages import AIMessage

                error_msg = AIMessage(
                    content="I couldn't find a relevant Confluence page matching your query. Please try a different search term."
                )
                return {"messages": [error_msg]}

            # Extract number
            import re

            numbers = re.findall(r"\d+", selection_text)
            if numbers:
                selected_idx = int(numbers[0]) - 1
                if 0 <= selected_idx < len(search_results):
                    selected_page = search_results[selected_idx]
                    logger.info(
                        f"Selected page: {selected_page.get('title', 'Unknown')}"
                    )
                    logger.info("=" * 60)

                    analysis_context["selected_confluence_page"] = selected_page
                    return {"analysis_context": analysis_context}
        except Exception as e:
            logger.error(f"Error parsing page selection: {e}")

        # Default to first result if parsing fails
        selected_page = search_results[0]
        logger.info(f"Defaulted to first page: {selected_page.get('title', 'Unknown')}")
        logger.info("=" * 60)

        analysis_context["selected_confluence_page"] = selected_page
        return {"analysis_context": analysis_context}

    async def get_confluence_page(state: AgentState):
        """Fetch the full content of the selected Confluence page."""
        logger.info("=" * 60)
        logger.info("NODE: get_confluence_page")

        analysis_context = state.get("analysis_context", {})
        selected_page = analysis_context.get("selected_confluence_page")

        if not selected_page:
            from langchain_core.messages import AIMessage

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
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content="Selected page has no ID. Cannot fetch content."
            )
            return {"messages": [error_msg]}

        # Find Confluence get page tool
        confluence_tool = None

        # Debug: Log all available tool names
        available_tool_names = [tool.name for tool in tools]
        logger.info(f"Available tools for get_page: {available_tool_names}")

        for tool in tools:
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
            for tool in tools:
                tool_name = tool.name.lower()
                if ("get" in tool_name or "fetch" in tool_name) and "page" in tool_name:
                    confluence_tool = tool
                    logger.info(f"Found get tool (alternative): {tool.name}")
                    break

        if not confluence_tool:
            from langchain_core.messages import AIMessage

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
                import json

                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Extract page content - adapt based on actual tool response format
            page_content = (
                result.get("content") or result.get("body") or result.get("text") or ""
            )
            page_title = result.get("title") or selected_page.get("title") or "Unknown"
            page_url = result.get("url") or selected_page.get("url") or ""

            logger.info(f"Fetched Confluence page: {page_title}")
            logger.info("=" * 60)

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
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(content=f"Failed to fetch Confluence page: {str(e)}")
            return {"messages": [error_msg]}

    def summarize_or_answer_from_page(state: AgentState):
        """Summarize or answer questions based on Confluence page content."""
        logger.info("=" * 60)
        logger.info("NODE: summarize_or_answer_from_page")

        analysis_context = state.get("analysis_context", {})
        page_content_data = analysis_context.get("confluence_page_content")

        if not page_content_data:
            from langchain_core.messages import AIMessage

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
        from langchain_core.messages import HumanMessage, SystemMessage

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

        response = llm.invoke(prompt_messages)

        # Format response with page reference
        from langchain_core.messages import AIMessage

        formatted_response = (
            f"{response.content}\n\n*Source: [{page_title}]({page_url})*"
        )
        answer_msg = AIMessage(content=formatted_response)

        logger.info("Generated summary/answer from Confluence page")
        logger.info("=" * 60)

        return {"messages": [answer_msg]}

    # Create the graph
    workflow = StateGraph(AgentState)

    # Create a custom async tool node wrapper for logging
    async def call_tools(state: AgentState):
        """Call tools with logging (async to support MCP tools)."""
        logger.info("=" * 60)
        logger.info("NODE: tools (ToolNode)")
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if (
            last_message
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            logger.info(f"Executing {len(last_message.tool_calls)} tool call(s):")
            for tool_call in last_message.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                tool_args = getattr(tool_call, "args", {})
                logger.info(f"  - Tool: {tool_name}")
                logger.debug(f"    Args: {tool_args}")

        # Use ToolNode to execute tools (async for MCP tools)
        tool_node = ToolNode(tools)
        result = await tool_node.ainvoke(state)

        # Log tool results
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "name"):
                    tool_name = getattr(msg, "name", "unknown")
                    logger.info(f"Tool result received from: {tool_name}")

        return result

    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("document_qa", document_qa_node)
    workflow.add_node("knowledge_enrichment", knowledge_enrichment_node)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    # Confluence Export subflow nodes
    workflow.add_node("ensure_analysis_context", ensure_analysis_context)
    workflow.add_node("build_confluence_page_draft", build_confluence_page_draft)
    workflow.add_node("decide_confluence_destination", decide_confluence_destination)
    workflow.add_node("create_confluence_page", create_confluence_page)
    workflow.add_node(
        "store_confluence_page_info", store_confluence_page_info_and_respond
    )

    # Confluence Read subflow nodes
    workflow.add_node("understand_confluence_query", understand_confluence_query)
    workflow.add_node("confluence_search", confluence_search_node)
    workflow.add_node("select_confluence_page", select_confluence_page)
    workflow.add_node("get_confluence_page", get_confluence_page)
    workflow.add_node("summarize_from_confluence", summarize_or_answer_from_page)

    # Set entry point
    workflow.set_entry_point("classify")

    # Route from classifier
    def route_after_classify(state: AgentState) -> str:
        """Route based on query classification and doc_action."""
        classification = state.get("query_classification", "DATA_ANALYSIS")
        doc_action = state.get("doc_action", "NONE")
        logger.info(
            f"Routing based on classification: {classification}, doc_action: {doc_action}"
        )

        # If user wants to read from Confluence, route to Confluence read subflow
        if doc_action == "FROM_CONFLUENCE":
            return "understand_confluence_query"

        # Otherwise, route based on classification as before
        if classification == "DOCUMENT_QA":
            return "document_qa"
        elif classification == "BOTH":
            return "knowledge_enrichment"
        else:  # DATA_ANALYSIS
            return "agent"

    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "document_qa": "document_qa",
            "knowledge_enrichment": "knowledge_enrichment",
            "agent": "agent",
            "understand_confluence_query": "understand_confluence_query",
        },
    )

    # Route from document_qa
    def route_after_document_qa(state: AgentState) -> str:
        """Route after document QA - check if tools are needed."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if (
            last_message
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            return "tools"
        return "end"

    workflow.add_conditional_edges(
        "document_qa",
        route_after_document_qa,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # Route from tools back to document_qa if we're in document QA mode
    def route_from_tools(state: AgentState) -> str:
        """Route from tools - check if we're in document QA mode."""
        classification = state.get("query_classification", "DATA_ANALYSIS")
        if classification == "DOCUMENT_QA":
            return "document_qa"
        else:
            return "agent"

    workflow.add_conditional_edges(
        "tools",
        route_from_tools,
        {
            "document_qa": "document_qa",
            "agent": "agent",
        },
    )

    # Route from knowledge_enrichment to agent
    workflow.add_edge("knowledge_enrichment", "agent")

    # Add conditional edges from agent
    def route_after_agent(state: AgentState) -> str:
        """Route after agent - check if we need to continue, end, or export to Confluence."""
        doc_action = state.get("doc_action", "NONE")

        # Check if we should continue with tools
        if should_continue(state) == "continue":
            return "continue"  # Return the key, not the destination

        # If analysis is done and user wants to export to Confluence, route to export subflow
        if doc_action == "FROM_ANALYSIS":
            return "ensure_analysis_context"

        # Otherwise, end
        return "end"

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "continue": "tools",
            "end": END,
            "ensure_analysis_context": "ensure_analysis_context",
        },
    )

    # Confluence Export subflow routing
    workflow.add_edge("ensure_analysis_context", "build_confluence_page_draft")
    workflow.add_edge("build_confluence_page_draft", "decide_confluence_destination")
    workflow.add_edge("decide_confluence_destination", "create_confluence_page")
    workflow.add_edge("create_confluence_page", "store_confluence_page_info")
    workflow.add_edge("store_confluence_page_info", END)

    # Confluence Read subflow routing
    workflow.add_edge("understand_confluence_query", "confluence_search")
    workflow.add_edge("confluence_search", "select_confluence_page")

    # Route from select_confluence_page - if meta-question, we already responded, so end
    # Otherwise, continue to get page content
    def route_after_select_page(state: AgentState) -> str:
        """Route after selecting page - check if we already responded (meta-question) or need to fetch content."""
        analysis_context = state.get("analysis_context", {})
        is_meta_question = analysis_context.get("confluence_is_meta_question", False)
        selected_page = analysis_context.get("selected_confluence_page")

        # If it's a meta-question, we already responded in select_confluence_page, so end
        if is_meta_question:
            return "end"

        # If no page was selected, we already responded with an error, so end
        if not selected_page:
            return "end"

        # Otherwise, fetch the page content
        return "get_confluence_page"

    workflow.add_conditional_edges(
        "select_confluence_page",
        route_after_select_page,
        {
            "end": END,
            "get_confluence_page": "get_confluence_page",
        },
    )

    workflow.add_edge("get_confluence_page", "summarize_from_confluence")
    workflow.add_edge("summarize_from_confluence", END)

    # Tools routing is handled by conditional edges above

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("Agent graph created successfully")
    return app


# Cache for the compiled graph to avoid recreating it
_graph_cache = None


# LangGraph Server entry point - can be async or sync
async def graph():
    """
    Entry point for LangGraph Server.
    This function is called by `langgraph dev` to get the graph.

    Returns:
        Compiled LangGraph StateGraph
    """
    global _graph_cache

    # Cache the graph to avoid recreating it on every call
    if _graph_cache is None:
        _graph_cache = await create_agent()

    return _graph_cache


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine whether to continue or end based on the last message."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # Otherwise, end
    return "end"


def generate_workflow_diagram(
    app, output_path: str | Path = "agent_workflow.png", format: str = "png"
) -> str:
    """
    Generate a Mermaid diagram of the agent workflow from the compiled graph.

    This method automatically extracts the graph structure and generates a visual
    representation. If the workflow changes, running this method will automatically
    reflect those changes in the diagram.

    Args:
        app: The compiled LangGraph application (returned from create_agent)
        output_path: Path where to save the diagram file. Defaults to "agent_workflow.png"
        format: Output format. Can be "png", "svg", or "mermaid". Defaults to "png"

    Returns:
        Path to the generated diagram file

    Example:
        >>> from src.langgraph_server.graph import create_agent, generate_workflow_diagram
        >>> app = await create_agent()
        >>> diagram_path = generate_workflow_diagram(app, "workflow.png")
        >>> print(f"Diagram saved to {diagram_path}")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        graph = app.get_graph()

        if format.lower() == "mermaid":
            # Get Mermaid code
            mermaid_code = graph.draw_mermaid()
            output_path = output_path.with_suffix(".mmd")
            output_path.write_text(mermaid_code, encoding="utf-8")
            logger.info(f"Mermaid diagram saved to {output_path}")
            return str(output_path)
        elif format.lower() == "png":
            # Generate PNG
            graph.draw_mermaid_png(output_file_path=str(output_path))
            logger.info(f"Workflow diagram (PNG) saved to {output_path}")
            return str(output_path)
        elif format.lower() == "svg":
            # Generate SVG
            graph.draw_mermaid_svg(output_file_path=str(output_path))
            logger.info(f"Workflow diagram (SVG) saved to {output_path}")
            return str(output_path)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Supported formats: png, svg, mermaid"
            )

    except AttributeError as e:
        logger.error(
            f"Error generating diagram: {e}. "
            "Make sure you're passing a compiled LangGraph application."
        )
        raise
    except Exception as e:
        logger.error(f"Error generating workflow diagram: {e}")
        raise
