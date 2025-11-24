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
    CODE_GENERATION_PROMPT,
    COMBINED_CLASSIFICATION_PROMPT,
    CONFLUENCE_QUERY_UNDERSTANDING_PROMPT,
    DOCUMENT_QA_PROMPT,
    KNOWLEDGE_ENRICHMENT_PROMPT,
    VERIFIER_PROMPT,
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
    verification_retry_count: NotRequired[int]  # Track verification retries
    query_classification: NotRequired[Literal["DOCUMENT_QA", "DATA_ANALYSIS", "BOTH"]]
    knowledge_context: NotRequired[str]
    doc_action: NotRequired[Literal["NONE", "FROM_ANALYSIS", "FROM_CONFLUENCE"]]
    confluence_page_id: NotRequired[str]
    confluence_page_url: NotRequired[str]
    analysis_context: NotRequired[
        dict
    ]  # Stores recent analysis results for Confluence export
    verification_result: NotRequired[dict]  # Stores verification result


def extract_content_text(content):
    """Extract text from content, handling both string and list formats.

    Anthropic Claude returns content as a list like [{"type": "text", "text": "..."}]
    while other models return a plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from Anthropic's content blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Anthropic format: {"type": "text", "text": "..."}
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
                # Also handle direct string values in dict
                elif "text" in item:
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    # Fallback: convert to string
    return str(content)


async def create_agent():
    """
    Create the data analysis agent graph.

    LLM configuration is loaded from environment variables:
    - CHAT_NODE prefix for the main reasoning agent
    - CODING_NODE prefix (optional) for the code generation agent

    For example:
        CHAT_NODE__llm_model_provider=openai
        CHAT_NODE__llm_model_name=gpt-4o
        CHAT_NODE__temperature=0.1
        CHAT_NODE__api_key=sk-...

        CODING_NODE__llm_model_provider=openai
        CODING_NODE__llm_model_name=gpt-4o
        CODING_NODE__temperature=0.1
        CODING_NODE__api_key=sk-...

    If CODING_NODE is not configured, the main LLM (CHAT_NODE) will be used for code generation.

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

    # Initialize the main reasoning LLM from settings
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

    # Bind tools to the main LLM
    # For Qwen and GPT-OSS models, use tool_choice for stable behavior
    tool_choice = None
    chat_model_name_lower = settings.chat_llm.llm_model_name.lower()
    is_chat_qwen = "qwen" in chat_model_name_lower
    is_chat_gpt_oss = (
        "gpt-oss" in chat_model_name_lower or "gpt_oss" in chat_model_name_lower
    )

    if settings.chat_llm.llm_model_provider == LLMProvider.QWEN_OLLAMA:
        tool_choice = getattr(settings.chat_llm, "tool_choice", "required")
        logger.info(f"Using tool_choice='{tool_choice}' for QwenOllama")
    elif settings.chat_llm.llm_model_provider == LLMProvider.LOCAL and (
        is_chat_qwen or is_chat_gpt_oss
    ):
        tool_choice = getattr(settings.chat_llm, "tool_choice", "required")
        model_type = "Qwen" if is_chat_qwen else "GPT-OSS"
        logger.info(
            f"Using tool_choice='{tool_choice}' for {model_type} model ({settings.chat_llm.llm_model_name})"
        )

    if tool_choice:
        llm_with_tools = llm.bind_tools(tools, tool_choice=tool_choice)
    else:
        llm_with_tools = llm.bind_tools(tools)

    # Filter tools for code generation - only run_analysis
    # get_dataset_schema is handled by the main agent to prevent loops
    coding_tools = [tool for tool in tools if tool.name in ["run_analysis"]]
    coding_tool_names = [tool.name for tool in coding_tools]
    logger.info("=" * 60)
    logger.info("Code generation tools (limited set): %d tools", len(coding_tools))
    for tool_name in coding_tool_names:
        logger.info("  - %s", tool_name)
    logger.info("=" * 60)

    # Initialize the coding LLM if configured, otherwise use main LLM
    # CRITICAL: Force tool_choice="required" for coding LLM to ensure it always makes tool calls
    # This prevents the model from outputting natural language explanations instead of tool calls
    # NOTE: Anthropic Claude doesn't support tool_choice="required", so we skip it for Anthropic
    if settings.coding_llm:
        logger.info(
            f"Using separate coding LLM: {settings.coding_llm.llm_model_name} "
            f"(provider: {settings.coding_llm.llm_model_provider})"
        )
        coding_llm = initialize_llm(settings.coding_llm)
        # Force tool_choice="required" for coding LLM to ensure tool calls are always made
        # Allow override via settings, but default to "required"
        # EXCEPTION: Anthropic doesn't support tool_choice="required"
        coding_tool_choice = getattr(settings.coding_llm, "tool_choice", "required")

        # Anthropic doesn't support tool_choice="required" - it interprets it as a tool name
        if settings.coding_llm.llm_model_provider == LLMProvider.ANTHROPIC:
            coding_tool_choice = (
                None  # Don't use tool_choice for Anthropic, rely on prompt
            )
            logger.info(
                "Anthropic provider detected - skipping tool_choice (Anthropic doesn't support 'required'). "
                "Relying on prompt to ensure tool calls."
            )
        else:
            # For Qwen models (QWEN_OLLAMA or LOCAL with Qwen/gpt-oss), always use "required"
            model_name_lower = settings.coding_llm.llm_model_name.lower()
            is_qwen_model = "qwen" in model_name_lower
            is_gpt_oss_model = (
                "gpt-oss" in model_name_lower or "gpt_oss" in model_name_lower
            )
            if settings.coding_llm.llm_model_provider in (
                LLMProvider.QWEN_OLLAMA,
                LLMProvider.LOCAL,
            ) and (is_qwen_model or is_gpt_oss_model):
                coding_tool_choice = "required"
                logger.info(
                    f"Detected {'Qwen' if is_qwen_model else 'GPT-OSS'} model: {settings.coding_llm.llm_model_name}, "
                    f"forcing tool_choice='required'"
                )

        logger.info(
            f"Using tool_choice={coding_tool_choice} for coding LLM "
            f"(provider: {settings.coding_llm.llm_model_provider})"
        )
        if coding_tool_choice is not None:
            coding_llm_with_tools = coding_llm.bind_tools(
                coding_tools, tool_choice=coding_tool_choice
            )
        else:
            # For Anthropic, bind tools without tool_choice and rely on prompt
            coding_llm_with_tools = coding_llm.bind_tools(coding_tools)
    else:
        logger.info(
            "No separate coding LLM configured, using main LLM for code generation"
        )
        coding_llm = llm
        # Force tool_choice="required" for code generation even when using main LLM
        # This ensures code generation always produces tool calls
        # EXCEPTION: Anthropic doesn't support tool_choice="required"
        if settings.chat_llm.llm_model_provider == LLMProvider.ANTHROPIC:
            coding_tool_choice = None
            logger.info(
                "Anthropic provider detected for main LLM - skipping tool_choice for code generation. "
                "Relying on prompt to ensure tool calls."
            )
            coding_llm_with_tools = llm.bind_tools(coding_tools)
        else:
            coding_tool_choice = "required"
            logger.info(
                f"Using tool_choice='{coding_tool_choice}' for code generation (using main LLM)"
            )
            coding_llm_with_tools = llm.bind_tools(
                coding_tools, tool_choice=coding_tool_choice
            )

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

        # If we have tool responses but no run_analysis yet, check if we should route to code generation
        # This is especially important for Qwen which might stop after list_datasets
        if has_tool_responses and not has_run_analysis:
            # Check if we have dataset information (from list_datasets or get_dataset_schema)
            has_dataset_info = False
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", "")
                    if tool_name in ["list_datasets", "get_dataset_schema"]:
                        has_dataset_info = True
                        break

            if has_dataset_info:
                # We have dataset info but no code generated yet - route to code generation
                logger.info(
                    "Dataset information available but no code generated - routing to code generation node"
                )
                # Return a special marker to indicate code generation is needed
                from langchain_core.messages import AIMessage

                routing_msg = AIMessage(
                    content="CODE_GENERATION_NEEDED: Dataset information has been gathered. Code generation is required to proceed with the analysis."
                )
                return {"messages": [routing_msg]}

            # Otherwise, inject a reminder
            from langchain_core.messages import SystemMessage

            reminder_msg = SystemMessage(
                content="REMINDER: You have received tool responses (e.g., from list_datasets or get_dataset_schema). "
                "These are INTERMEDIATE INFORMATION, not final answers. "
                "You MUST continue the workflow by routing to code generation to execute the actual data analysis. "
                "DO NOT stop and present tool responses as your final answer. "
                "The workflow is only complete after code has been generated and run_analysis has been executed."
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

    # Code generation node
    def code_generation_node(state: AgentState):
        """Generate Python code for data analysis using the coding LLM."""
        messages = state["messages"]
        logger.info("=" * 60)
        logger.info("NODE: code_generation")

        # Extract dataset IDs from tool responses
        import json

        from langchain_core.messages import SystemMessage, ToolMessage

        dataset_ids_found = set()
        schemas_found = {}  # Track which datasets have schema information
        schema_call_count = {}  # Track how many times get_dataset_schema was called per dataset

        # Look through messages for tool responses that contain dataset information
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "")

                # Extract dataset_id from get_dataset_schema responses
                if tool_name == "get_dataset_schema":
                    # Try to extract dataset_id from the tool call that preceded this response
                    # Look for the tool call in previous messages
                    for prev_msg in reversed(messages[: messages.index(msg)]):
                        if hasattr(prev_msg, "tool_calls") and prev_msg.tool_calls:
                            for tool_call in prev_msg.tool_calls:
                                if (
                                    getattr(tool_call, "name", "")
                                    == "get_dataset_schema"
                                ):
                                    args = getattr(tool_call, "args", {})
                                    if isinstance(args, dict):
                                        # Handle both direct args and nested payload
                                        dataset_id = args.get("dataset_id") or args.get(
                                            "payload", {}
                                        ).get("dataset_id")
                                        if dataset_id:
                                            dataset_ids_found.add(dataset_id)
                                            # Mark that we have schema for this dataset
                                            schemas_found[dataset_id] = True
                                            # Track how many times this dataset's schema was called
                                            schema_call_count[dataset_id] = (
                                                schema_call_count.get(dataset_id, 0) + 1
                                            )
                                            break

                # Extract dataset IDs from list_datasets responses
                elif tool_name == "list_datasets":
                    try:
                        # Try to parse the tool response content
                        content = getattr(msg, "content", "")
                        data = None

                        if isinstance(content, dict):
                            # Content is already a dict
                            data = content
                        elif isinstance(content, str):
                            # Try to parse as JSON
                            try:
                                data = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                pass

                        if isinstance(data, dict) and "datasets" in data:
                            for dataset in data["datasets"]:
                                if isinstance(dataset, dict) and "id" in dataset:
                                    dataset_ids_found.add(dataset["id"])
                    except Exception as e:
                        logger.debug(
                            f"Error extracting dataset IDs from list_datasets response: {e}"
                        )

        # Extract the user's original question for clarity
        user_question = None
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content if hasattr(msg, "content") else str(msg)
                break

        # Filter out routing messages that might confuse the model
        filtered_messages = []
        for msg in messages:
            # Skip the CODE_GENERATION_NEEDED routing message
            if (
                hasattr(msg, "content")
                and isinstance(msg.content, str)
                and "CODE_GENERATION_NEEDED" in msg.content
            ):
                continue
            filtered_messages.append(msg)

        # Add knowledge context if available
        prompt_messages = list(filtered_messages)
        knowledge_context = state.get("knowledge_context", "")

        # Build system message with context
        system_parts = []

        if (
            knowledge_context
            and knowledge_context.strip()
            and len(knowledge_context.strip()) > 10
        ):
            system_parts.append(
                f"Document knowledge context:\n{knowledge_context}\n\nUse this context to understand domain terms and map them to dataset columns when writing code."
            )

        # Add the user's question prominently if available
        if user_question:
            system_parts.append(
                f"USER'S QUESTION (THIS IS WHAT YOU NEED TO ANSWER):\n{user_question}\n"
            )

        # Add dataset IDs hint if found
        if dataset_ids_found:
            dataset_ids_list = sorted(list(dataset_ids_found))
            datasets_with_schema = [
                ds_id for ds_id in dataset_ids_list if schemas_found.get(ds_id, False)
            ]
            datasets_without_schema = [
                ds_id
                for ds_id in dataset_ids_list
                if not schemas_found.get(ds_id, False)
            ]

            schema_info = []
            if datasets_with_schema:
                schema_info.append(
                    f"Datasets with schema information (you can use these directly): {', '.join(datasets_with_schema)}"
                )
            if datasets_without_schema:
                schema_info.append(
                    f"CRITICAL - Datasets WITHOUT schema information: {', '.join(datasets_without_schema)}\n"
                    f"  - You MUST call get_dataset_schema() for these datasets BEFORE generating code\n"
                    f"  - Do NOT guess column names - this will cause KeyError\n"
                    f"  - Example: Call get_dataset_schema('{datasets_without_schema[0]}') first, then use the schema to write correct code"
                )

            schema_info_text = "\n".join(schema_info) if schema_info else ""
            system_parts.append(
                f"CRITICAL - Datasets identified in conversation:\n"
                f"The following dataset IDs have been mentioned or examined: {', '.join(dataset_ids_list)}\n"
                f"{schema_info_text}\n"
                f"When you generate code that uses these datasets (e.g., dfs['{dataset_ids_list[0]}']), "
                f"you MUST include ALL referenced dataset IDs in the dataset_ids parameter of run_analysis.\n"
                f"Example: If your code uses dfs['{dataset_ids_list[0]}'], then dataset_ids=['{dataset_ids_list[0]}']\n"
            )

            if datasets_without_schema:
                system_parts.append(
                    f"\n⚠️ WARNING: The main agent has not yet retrieved schema for these datasets: {', '.join(datasets_without_schema)}\n"
                    f"However, you MUST still generate code and call run_analysis. Use common column name patterns or check the conversation history for any schema hints.\n"
                    f"The main agent should have gathered this information, but proceed with code generation anyway."
                )
            else:
                system_parts.append(
                    "\n✅ YOU HAVE ALL SCHEMA INFORMATION. Generate code using the schema information from the conversation history and call run_analysis immediately."
                )

            # Add explicit instruction about using schema from conversation
            if datasets_with_schema:
                system_parts.append(
                    f"\n✅ Schema information is available in the conversation history for: {', '.join(datasets_with_schema)}\n"
                    f"Use the schema information from get_dataset_schema() responses in the conversation to generate correct code.\n"
                    f"Then call run_analysis with your generated code."
                )
        else:
            # Even if no dataset IDs found, we should still try to generate code
            system_parts.append(
                "NOTE: Generate code based on the user's question and available context from the conversation history.\n"
                "The main agent has gathered all necessary information. You MUST call run_analysis with your generated code.\n"
                "DO NOT refuse - generate code and execute it."
            )

        if system_parts:
            system_content = "\n\n".join(system_parts)
            prompt_messages.insert(0, SystemMessage(content=system_content))
            logger.info(
                f"Injected system context with {len(dataset_ids_found)} dataset ID(s): {sorted(dataset_ids_found)}"
            )

        prompt = CODE_GENERATION_PROMPT.invoke({"messages": prompt_messages})
        logger.info("Invoking coding LLM for code generation...")
        logger.info(f"Available coding tools: {[tool.name for tool in coding_tools]}")
        logger.info(f"Tool choice setting: {coding_tool_choice}")
        logger.info(f"Number of messages in prompt: {len(prompt.messages)}")
        response = coding_llm_with_tools.invoke(prompt.messages)

        # Handle Anthropic Claude's content format (list of content blocks)
        # Anthropic returns content as a list like [{"type": "text", "text": "..."}]
        # while other models return a plain string
        def extract_content_text(content):
            """Extract text from content, handling both string and list formats."""
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Extract text from Anthropic's content blocks
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        # Anthropic format: {"type": "text", "text": "..."}
                        if item.get("type") == "text" and "text" in item:
                            text_parts.append(str(item["text"]))
                        # Also handle direct string values in dict
                        elif "text" in item:
                            text_parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        text_parts.append(item)
                return " ".join(text_parts)
            # Fallback: convert to string
            return str(content)

        response_content_text = extract_content_text(getattr(response, "content", None))

        # Validate response - check if model output natural language instead of tool calls
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
        has_content = response_content_text and response_content_text.strip()

        # Detect refusal patterns in the response
        refusal_keywords = [
            "cannot",
            "can't",
            "unable",
            "refuse",
            "not able",
            "don't have",
            "need more",
            "insufficient",
            "lack of",
            "missing information",
            "not enough",
            "requires more",
            "cannot proceed",
            "unable to",
        ]
        is_refusal = False
        if has_content:
            content_lower = response_content_text.lower()
            is_refusal = any(keyword in content_lower for keyword in refusal_keywords)

        if not has_tool_calls and has_content:
            content_preview = (
                response_content_text[:200]
                if len(response_content_text) > 200
                else response_content_text
            )

            if is_refusal:
                logger.error(
                    f"Code generation LLM REFUSED to make tool calls. "
                    f"Refusal message: {content_preview}..."
                )
                logger.error(
                    "This is likely because the model doesn't have enough context or information. "
                    "Attempting to provide additional context and retry..."
                )

                # Try to provide more context and retry once
                # Add a more explicit instruction with example
                from langchain_core.messages import SystemMessage

                retry_system_msg = SystemMessage(
                    content=f"""CRITICAL: You MUST make a tool call. Even if you're uncertain, you MUST still call run_analysis with the best code you can generate.

The previous response was a refusal, which is not allowed. You MUST make a tool call.

If you're missing information:
1. Use the dataset IDs that were mentioned in the conversation: {sorted(list(dataset_ids_found)) if dataset_ids_found else "none found"}
2. Generate code based on the user's question and available context
3. Even if the code might not be perfect, you MUST call run_analysis

EXAMPLE - You MUST do this:
Call run_analysis with:
- code: "result_df = dfs['dataset_id'].head(10)"  # Replace with actual analysis code
- dataset_ids: ['dataset_id']  # Use actual dataset IDs from conversation

DO NOT refuse. DO NOT explain why you can't. Just make the tool call."""
                )

                # Insert the retry message before the last system message (if any) or at the beginning
                retry_prompt_messages = list(prompt_messages)
                # Find the last SystemMessage and insert after it, or insert at the beginning
                insert_idx = 0
                for i, msg in enumerate(retry_prompt_messages):
                    if isinstance(msg, SystemMessage):
                        insert_idx = i + 1
                retry_prompt_messages.insert(insert_idx, retry_system_msg)

                retry_prompt = CODE_GENERATION_PROMPT.invoke(
                    {"messages": retry_prompt_messages}
                )
                logger.info("Retrying code generation with additional context...")
                response = coding_llm_with_tools.invoke(retry_prompt.messages)

                # Re-check after retry
                # Re-extract content text after retry (may be list for Anthropic)
                response_content_text = extract_content_text(
                    getattr(response, "content", None)
                )
                has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
                has_content = response_content_text and response_content_text.strip()
                if has_content:
                    content_lower = response_content_text.lower()
                    is_refusal = any(
                        keyword in content_lower for keyword in refusal_keywords
                    )

                if not has_tool_calls and is_refusal:
                    logger.error(
                        "Code generation LLM still refused after retry with additional context. "
                        "This indicates a deeper issue - the model may not have sufficient information "
                        "or there may be a configuration problem."
                    )
                    # Log the full refusal message for debugging
                    logger.error(f"Full refusal message: {response_content_text}")
                    # Log available context for debugging
                    logger.error(
                        f"Available dataset IDs from conversation: {sorted(list(dataset_ids_found)) if dataset_ids_found else 'none'}"
                    )
                    logger.error(
                        f"Number of messages in context: {len(prompt_messages)}"
                    )
            else:
                logger.error(
                    f"Code generation LLM output natural language instead of tool calls. "
                    f"Content preview: {content_preview}..."
                )
                logger.error(
                    "This indicates the model did not follow instructions to make tool calls. "
                    "Check tool_choice setting and prompt."
                )

            # Try to extract code from the natural language response and create a tool call
            # This handles both refusals and other natural language responses
            import re

            code_match = re.search(
                r"```(?:python)?\s*\n(.*?)```", response_content_text, re.DOTALL
            )
            if code_match:
                extracted_code = code_match.group(1).strip()
                logger.warning(
                    "Attempting to extract code from natural language response"
                )
                # Try to extract dataset IDs from the extracted code
                pattern = r"dfs\[['\"]([^'\"]+)['\"]\]"
                code_dataset_ids = set(re.findall(pattern, extracted_code))

                if code_dataset_ids:
                    # Create a synthetic tool call
                    from langchain_core.messages.tool import ToolCall

                    tool_call = ToolCall(
                        name="run_analysis",
                        args={
                            "code": extracted_code,
                            "dataset_ids": sorted(list(code_dataset_ids)),
                        },
                        id="extracted_from_natural_language",
                    )
                    response.tool_calls = [tool_call]
                    response.content = ""  # Clear the natural language content
                    logger.warning(
                        f"Extracted code and created synthetic tool call with dataset_ids: {sorted(code_dataset_ids)}"
                    )
                else:
                    # Fallback to dataset_ids from conversation
                    if dataset_ids_found:
                        from langchain_core.messages.tool import ToolCall

                        tool_call = ToolCall(
                            name="run_analysis",
                            args={
                                "code": extracted_code,
                                "dataset_ids": sorted(list(dataset_ids_found)),
                            },
                            id="extracted_from_natural_language",
                        )
                        response.tool_calls = [tool_call]
                        response.content = ""
                        logger.warning(
                            f"Extracted code and used dataset_ids from conversation: {sorted(dataset_ids_found)}"
                        )
                    else:
                        logger.error(
                            "Could not extract code or determine dataset_ids. "
                            "Response will be passed through but may cause errors."
                        )
                        # If this was a refusal and we couldn't extract code, create a minimal tool call
                        # to prevent the workflow from completely failing
                        if is_refusal and dataset_ids_found:
                            logger.warning(
                                "Creating minimal tool call with placeholder code to prevent workflow failure. "
                                "This is a fallback - the code may need to be fixed by the main agent."
                            )
                            from langchain_core.messages.tool import ToolCall

                            # Create a minimal code that at least loads the dataset
                            placeholder_code = f"# Placeholder code - model refused to generate proper code\nresult_df = dfs['{list(dataset_ids_found)[0]}'].head(1)"
                            tool_call = ToolCall(
                                name="run_analysis",
                                args={
                                    "code": placeholder_code,
                                    "dataset_ids": sorted(list(dataset_ids_found)),
                                },
                                id="fallback_after_refusal",
                            )
                            response.tool_calls = [tool_call]
                            response.content = ""  # Clear the refusal message
                            logger.warning(
                                f"Created fallback tool call with dataset_ids: {sorted(list(dataset_ids_found))}"
                            )

        # Log tool calls if any and validate/fix them
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(
                f"Coding LLM requested {len(response.tool_calls)} tool call(s):"
            )
            import re

            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                logger.info(f"  - {tool_name}")

                # Validate and fix run_analysis tool calls
                if tool_name == "run_analysis":
                    args = getattr(tool_call, "args", {})
                    if not isinstance(args, dict):
                        args = {}

                    # Check if dataset_ids is missing
                    if "dataset_ids" not in args or not args.get("dataset_ids"):
                        code = args.get("code", "")
                        if code:
                            # Extract dataset IDs from code using regex
                            # Look for patterns like dfs['dataset_id'] or dfs["dataset_id"]
                            pattern = r"dfs\[['\"]([^'\"]+)['\"]\]"
                            code_dataset_ids = set(re.findall(pattern, code))

                            # Also check for code aliases (df_jpm_patients, etc.)
                            # These are less reliable, so we'll prioritize dfs['id'] patterns

                            if code_dataset_ids:
                                args["dataset_ids"] = sorted(list(code_dataset_ids))
                                tool_call.args = args
                                logger.warning(
                                    f"Fixed missing dataset_ids in run_analysis call. "
                                    f"Extracted from code: {args['dataset_ids']}"
                                )
                            elif dataset_ids_found:
                                # Fallback to dataset IDs found in conversation
                                args["dataset_ids"] = sorted(list(dataset_ids_found))
                                tool_call.args = args
                                logger.warning(
                                    f"Fixed missing dataset_ids in run_analysis call. "
                                    f"Using dataset IDs from conversation: {args['dataset_ids']}"
                                )
                            else:
                                logger.error(
                                    "run_analysis call is missing dataset_ids and could not be extracted from code or conversation."
                                )
        else:
            logger.warning(
                "Coding LLM response has no tool calls - code generation may have failed"
            )

        logger.info("Code generation response generated")
        return {"messages": [response]}

    # Verifier node
    def verify_response(state: AgentState):
        """Verify if the agent's response is sufficient to answer the user's query."""
        logger.info("=" * 60)
        logger.info("NODE: verify_response")

        messages = state["messages"]
        query_classification = state.get("query_classification", "DATA_ANALYSIS")
        verification_retry_count = state.get("verification_retry_count", 0)

        # Maximum retries to prevent infinite loops
        MAX_VERIFICATION_RETRIES = 3
        if verification_retry_count >= MAX_VERIFICATION_RETRIES:
            logger.warning(
                f"Maximum verification retries ({MAX_VERIFICATION_RETRIES}) reached. Accepting response."
            )
            return {
                "verification_result": {
                    "is_sufficient": True,
                    "reason": "Maximum retries reached",
                    "feedback": "",
                }
            }

        # Get the original user query
        user_query = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content if hasattr(msg, "content") else str(msg)
                break

        if not user_query:
            # No user query found, accept the response
            logger.warning("No user query found, accepting response")
            return {
                "verification_result": {
                    "is_sufficient": True,
                    "reason": "No user query found",
                    "feedback": "",
                }
            }

        # Check if run_analysis was called (for DATA_ANALYSIS queries)
        has_run_analysis = False
        has_tool_calls = False

        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    has_tool_calls = True
                    for tool_call in msg.tool_calls:
                        tool_name = getattr(tool_call, "name", "")
                        if tool_name == "run_analysis":
                            has_run_analysis = True
                            break
                break

        # Check for run_analysis in tool messages
        if not has_run_analysis:
            for msg in reversed(messages):
                if hasattr(msg, "name") and msg.name == "run_analysis":
                    has_run_analysis = True
                    break

        # For DATA_ANALYSIS queries, check if run_analysis was called
        if query_classification in ["DATA_ANALYSIS", "BOTH"]:
            if not has_run_analysis:
                # Check if we only have list_datasets or get_dataset_schema calls
                only_listing_tools = True
                for msg in reversed(messages):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = getattr(tool_call, "name", "")
                            if tool_name not in ["list_datasets", "get_dataset_schema"]:
                                only_listing_tools = False
                                break
                        if not only_listing_tools:
                            break

                if only_listing_tools and has_tool_calls:
                    logger.warning(
                        "Verification failed: Only listing tools called, no run_analysis"
                    )
                    return {
                        "verification_result": {
                            "is_sufficient": False,
                            "reason": "Only dataset listing tools were called, but no actual analysis was performed",
                            "feedback": "You only listed datasets or examined schemas but did not execute the analysis. You MUST call run_analysis to perform the actual data analysis and answer the user's question. The workflow is not complete until you have executed run_analysis and received actual results.",
                        },
                        "verification_retry_count": verification_retry_count + 1,
                    }

        # Use LLM to verify the response
        # Build context for verifier
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        # Get recent conversation context (last few messages)
        # Filter out ToolMessage objects to avoid pairing issues with OpenAI/GPT models
        # Tool messages must be immediately after AI messages with tool_calls, which is hard to guarantee
        # Instead, we'll include tool results as text summaries in the system message
        recent_messages_raw = messages[-10:]  # Last 10 messages for context
        recent_messages = []
        tool_results_summary = []

        for msg in recent_messages_raw:
            if isinstance(msg, ToolMessage):
                # Convert tool messages to text summaries to avoid pairing issues
                tool_name = getattr(msg, "name", "tool")
                tool_content = extract_content_text(getattr(msg, "content", ""))
                tool_results_summary.append(f"[{tool_name}]: {tool_content[:300]}")
            elif (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                # AI message with tool_calls - convert to text to avoid needing tool messages
                ai_content = extract_content_text(getattr(msg, "content", ""))
                if ai_content:
                    recent_messages.append(
                        HumanMessage(content=f"[AI called tools]: {ai_content[:500]}")
                    )
                else:
                    # No content, just tool calls - create a summary
                    tool_names = [getattr(tc, "name", "tool") for tc in msg.tool_calls]
                    recent_messages.append(
                        HumanMessage(
                            content=f"[AI called tools: {', '.join(tool_names)}]"
                        )
                    )
            else:
                # Human messages and AI messages without tool_calls - keep as-is
                recent_messages.append(msg)

        # Add tool results summary to system message if any were filtered
        system_content = f"User's original query: {user_query}\n\nQuery classification: {query_classification}\n\nHas run_analysis been called: {has_run_analysis}\n\nReview the conversation and determine if the response adequately answers the user's question."
        if tool_results_summary:
            system_content += "\n\nTool Results Summary:\n" + "\n".join(
                tool_results_summary
            )

        # Create verification prompt
        verification_messages = [
            SystemMessage(content=system_content)
        ] + recent_messages

        prompt = VERIFIER_PROMPT.invoke({"messages": verification_messages})
        logger.info("Invoking verifier LLM...")

        # Try up to 2 times to get valid JSON
        max_parse_attempts = 2
        verification_data = None
        response_text = None

        try:
            for attempt in range(max_parse_attempts):
                if attempt > 0:
                    logger.warning(
                        f"Retrying verifier JSON parsing (attempt {attempt + 1}/{max_parse_attempts})"
                    )
                    # Add a more explicit instruction for retry
                    retry_system_msg = SystemMessage(
                        content="CRITICAL: You MUST output ONLY a valid JSON object. No markdown, no code blocks, no text before or after. Just the JSON object starting with { and ending with }."
                    )
                    retry_messages = [retry_system_msg] + verification_messages[
                        1:
                    ]  # Keep the original system message context
                    retry_prompt = VERIFIER_PROMPT.invoke({"messages": retry_messages})
                    response = llm_json.invoke(retry_prompt.messages)
                else:
                    response = llm_json.invoke(prompt.messages)

                response_text = extract_content_text(getattr(response, "content", None))
                if response_text:
                    response_text = response_text.strip()

                # Log the raw response for debugging (only on first attempt to avoid spam)
                if attempt == 0:
                    logger.info(
                        f"Verifier raw response (first 500 chars): {response_text[:500]}"
                    )

                # Try to extract JSON from markdown code blocks if present
                import re

                json_match = re.search(
                    r"```(?:json)?\s*\n?({.*?})\n?```", response_text, re.DOTALL
                )
                if json_match:
                    response_text = json_match.group(1).strip()
                    logger.info("Extracted JSON from markdown code block")

                # Also try to extract JSON object directly if wrapped in text
                if not json_match:
                    # More robust pattern to find JSON object
                    json_match = re.search(
                        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\"is_sufficient\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
                        response_text,
                        re.DOTALL,
                    )
                    if json_match:
                        response_text = json_match.group(0).strip()
                        logger.info("Extracted JSON object from text")

                # Try to find JSON object by looking for the structure
                if not json_match:
                    # Look for JSON object that contains is_sufficient
                    brace_start = response_text.find("{")
                    if brace_start >= 0:
                        # Try to find matching closing brace
                        brace_count = 0
                        for i in range(brace_start, len(response_text)):
                            if response_text[i] == "{":
                                brace_count += 1
                            elif response_text[i] == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    potential_json = response_text[brace_start : i + 1]
                                    if '"is_sufficient"' in potential_json:
                                        response_text = potential_json
                                        logger.info(
                                            "Extracted JSON by finding matching braces"
                                        )
                                        break

                # Parse JSON response
                try:
                    verification_data = json.loads(response_text)
                    logger.info("Successfully parsed verification JSON")
                    break  # Success, exit retry loop
                except (ValueError, json.JSONDecodeError, TypeError) as parse_error:
                    if attempt < max_parse_attempts - 1:
                        logger.warning(
                            f"JSON parse failed on attempt {attempt + 1}: {parse_error}. Will retry..."
                        )
                        continue
                    else:
                        # Last attempt failed, will handle in except block below
                        raise

            # If we successfully parsed JSON, process it
            if verification_data is not None:
                is_sufficient = verification_data.get("is_sufficient", True)
                reason = verification_data.get("reason", "")
                feedback = verification_data.get("feedback", "")

                logger.info(f"Verification result: is_sufficient={is_sufficient}")
                logger.info(f"Reason: {reason}")
                if feedback:
                    logger.info(f"Feedback: {feedback}")

                # If verification failed, add feedback message to guide the agent
                result_updates = {
                    "verification_result": {
                        "is_sufficient": is_sufficient,
                        "reason": reason,
                        "feedback": feedback,
                    },
                    "verification_retry_count": (
                        verification_retry_count + 1 if not is_sufficient else 0
                    ),
                }

                if not is_sufficient and feedback:
                    from langchain_core.messages import SystemMessage

                    # Check if feedback message already exists to avoid duplicates
                    has_feedback = False
                    for msg in messages:
                        if (
                            isinstance(msg, SystemMessage)
                            and "VERIFICATION FEEDBACK" in msg.content
                        ):
                            has_feedback = True
                            break

                    if not has_feedback:
                        # Add feedback message to guide the agent
                        feedback_msg = SystemMessage(
                            content=f"VERIFICATION FEEDBACK: {feedback}\n\nYou must replan and continue the analysis. Do not stop until you have completed the full workflow."
                        )
                        # Add feedback to messages
                        current_messages = list(messages)
                        current_messages.insert(0, feedback_msg)
                        result_updates["messages"] = current_messages

                return result_updates

            # If we get here, JSON parsing failed after all retries
            # This should not happen due to the raise in the except block, but handle it anyway
            raise ValueError("Failed to parse JSON after all retry attempts")

        except (ValueError, json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"Failed to parse verification JSON response after all attempts: {e}"
            )
            if response_text:
                logger.error(f"Full response text: {response_text}")

            # Try to extract meaningful information from the response even if JSON parsing fails
            is_sufficient_heuristic = True
            reason_heuristic = "Could not parse verification response"

            # Try to infer from response text if available
            if response_text:
                response_lower = response_text.lower()
                if any(
                    word in response_lower
                    for word in [
                        "insufficient",
                        "not sufficient",
                        "missing",
                        "incomplete",
                        "failed",
                    ]
                ):
                    is_sufficient_heuristic = False
                    reason_heuristic = (
                        "Verifier indicated response is insufficient (parsed from text)"
                    )
                elif any(
                    word in response_lower
                    for word in ["sufficient", "complete", "adequate", "answered"]
                ):
                    is_sufficient_heuristic = True
                    reason_heuristic = (
                        "Verifier indicated response is sufficient (parsed from text)"
                    )

            # Fallback: check if run_analysis was called
            if (
                query_classification in ["DATA_ANALYSIS", "BOTH"]
                and not has_run_analysis
            ):
                return {
                    "verification_result": {
                        "is_sufficient": False,
                        "reason": "run_analysis was not called - analysis incomplete",
                        "feedback": "You must call run_analysis to perform the actual data analysis. The workflow is not complete until you have executed run_analysis and received actual results.",
                    },
                    "verification_retry_count": verification_retry_count + 1,
                }
            else:
                # Use heuristic if available, otherwise default based on run_analysis status
                if has_run_analysis:
                    return {
                        "verification_result": {
                            "is_sufficient": is_sufficient_heuristic,
                            "reason": f"{reason_heuristic}. run_analysis was called, so assuming response may be sufficient.",
                            "feedback": ""
                            if is_sufficient_heuristic
                            else "Please verify the response addresses the user's question completely.",
                        }
                    }
                else:
                    return {
                        "verification_result": {
                            "is_sufficient": is_sufficient_heuristic,
                            "reason": f"{reason_heuristic}. Unable to verify due to parsing error.",
                            "feedback": ""
                            if is_sufficient_heuristic
                            else "Please verify the response addresses the user's question completely.",
                        }
                    }

        logger.info("=" * 60)

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

        response_content = extract_content_text(getattr(response, "content", None))
        formatted_response = (
            f"{response_content}\n\n*Source: [{page_title}]({page_url})*"
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

        # Log tool results and detect plot generation
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "name"):
                    tool_name = getattr(msg, "name", "unknown")
                    logger.info(f"Tool result received from: {tool_name}")

                    # Detect plot generation from run_analysis results
                    if tool_name == "run_analysis" and hasattr(msg, "content"):
                        import json

                        try:
                            content = msg.content
                            if isinstance(content, str):
                                # Try to parse as JSON
                                try:
                                    result_data = json.loads(content)
                                except (json.JSONDecodeError, TypeError):
                                    # If not JSON, try to extract from string
                                    result_data = None

                                if result_data and isinstance(result_data, dict):
                                    plot_path = result_data.get("plot_path")
                                    plot_valid = result_data.get("plot_valid")

                                    if plot_path:
                                        logger.info("=" * 60)
                                        logger.info("PLOT DETECTED:")
                                        logger.info(f"  Plot path: {plot_path}")
                                        logger.info(f"  Plot valid: {plot_valid}")
                                        if plot_valid:
                                            logger.info(
                                                "  ✅ Plot was successfully generated and validated"
                                            )
                                        else:
                                            validation_msg = result_data.get(
                                                "plot_validation_message", "Unknown"
                                            )
                                            logger.warning(
                                                f"  ⚠️  Plot was generated but validation failed: {validation_msg}"
                                            )
                                        logger.info("=" * 60)
                                    else:
                                        logger.info(
                                            "No plot was generated in this run_analysis call"
                                        )
                        except Exception as e:
                            logger.debug(f"Error checking for plot in tool result: {e}")

        return result

    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("document_qa", document_qa_node)
    workflow.add_node("knowledge_enrichment", knowledge_enrichment_node)
    workflow.add_node("agent", call_model)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("tools", call_tools)
    workflow.add_node("verifier", verify_response)

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

    # Route from code_generation to tools
    workflow.add_edge("code_generation", "tools")

    # Route from tools back to appropriate node
    def route_from_tools(state: AgentState) -> str:
        """Route from tools - check context and determine next step."""
        classification = state.get("query_classification", "DATA_ANALYSIS")

        # If document QA mode, route back to document_qa
        if classification == "DOCUMENT_QA":
            return "document_qa"

        # Check recent tool calls to understand context
        messages = state["messages"]
        last_tool_message = None

        # Find the last tool message
        for msg in reversed(messages):
            if hasattr(msg, "name") and msg.name:
                last_tool_message = msg
                break

        # Note: get_dataset_schema is no longer available to code_generation node
        # It's only available to the main agent node to prevent loops
        # If get_dataset_schema was called, it was by the main agent, so route back to agent
        if last_tool_message and hasattr(last_tool_message, "name"):
            if last_tool_message.name == "get_dataset_schema":
                # get_dataset_schema is only called by main agent now
                # Route back to agent to continue gathering information or route to code generation
                logger.info(
                    "get_dataset_schema called by main agent, routing back to agent"
                )
                return "agent"

        # If run_analysis failed and we have error, might need to retry code generation
        if last_tool_message and hasattr(last_tool_message, "name"):
            if last_tool_message.name == "run_analysis":
                # Check if there's an error in the result
                if hasattr(last_tool_message, "content"):
                    content = last_tool_message.content
                    if isinstance(content, str):
                        # Check for error indicators
                        if (
                            '"error"' in content.lower()
                            or '"success": false' in content.lower()
                        ):
                            # Check if we should retry code generation or go back to agent
                            # For now, route back to agent to analyze the error
                            logger.info(
                                "run_analysis failed, routing back to agent for error analysis"
                            )
                            return "agent"

        # Default: route back to agent for next reasoning step
        return "agent"

    workflow.add_conditional_edges(
        "tools",
        route_from_tools,
        {
            "document_qa": "document_qa",
            "agent": "agent",
            "code_generation": "code_generation",
        },
    )

    # Route from knowledge_enrichment to agent
    workflow.add_edge("knowledge_enrichment", "agent")

    # Add conditional edges from agent
    def route_after_agent(state: AgentState) -> str:
        """Route after agent - check if we need code generation, tools, or verify."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        # Check if code generation is needed (marked by special message)
        if (
            last_message
            and hasattr(last_message, "content")
            and isinstance(last_message.content, str)
            and "CODE_GENERATION_NEEDED" in last_message.content
        ):
            logger.info("Routing to code generation node")
            return "code_generation"

        # Check if we should continue with tools
        if should_continue(state) == "continue":
            return "continue"  # Return the key, not the destination

        # Otherwise, route to verifier before ending or exporting
        # The verifier will handle routing to Confluence export if needed
        return "verifier"

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "code_generation": "code_generation",
            "continue": "tools",
            "verifier": "verifier",
        },
    )

    # Verifier routing
    def route_after_verifier(state: AgentState) -> str:
        """Route after verifier - check if response is sufficient."""
        verification_result = state.get("verification_result", {})
        is_sufficient = verification_result.get("is_sufficient", True)
        doc_action = state.get("doc_action", "NONE")

        if not is_sufficient:
            # Verification failed, route back to agent with feedback
            # (Feedback message is already added by verifier node)
            logger.info("Verification failed, routing back to agent with feedback")
            return "agent"
        else:
            # Verification passed
            # If analysis is done and user wants to export to Confluence, route to export subflow
            if doc_action == "FROM_ANALYSIS":
                return "ensure_analysis_context"
            # Otherwise, end
            return "end"

    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "agent": "agent",
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


def get_recursion_limit() -> int:
    """Get the appropriate recursion limit based on the configured model.

    All models use the same recursion limit (50).

    Returns:
        int: The recursion limit to use (50 for all models)
    """
    from .settings import LLMProvider, get_settings

    settings = get_settings()
    chat_model_name_lower = settings.chat_llm.llm_model_name.lower()

    # Check if using qwen or gpt-oss models
    is_qwen = "qwen" in chat_model_name_lower
    is_gpt_oss = (
        "gpt-oss" in chat_model_name_lower or "gpt_oss" in chat_model_name_lower
    )

    # Check provider
    is_qwen_provider = settings.chat_llm.llm_model_provider == LLMProvider.QWEN_OLLAMA
    is_local_with_qwen_or_gpt_oss = (
        settings.chat_llm.llm_model_provider == LLMProvider.LOCAL
        and (is_qwen or is_gpt_oss)
    )

    if is_qwen_provider or is_local_with_qwen_or_gpt_oss:
        logger.info(
            f"Using recursion_limit (50) for {'Qwen' if is_qwen else 'GPT-OSS'} model"
        )
        return 50

    # Default recursion limit for other models
    return 50


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
