"""LangGraph definition for the data analysis agent."""

import logging
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from .llm_utils import initialize_llm
from .mcp_tool_loader import MCPToolLoader
from .nodes import (
    AgentNode,
    BuildConfluencePageDraftNode,
    ClassifyQueryNode,
    CodeGenerationNode,
    ConfluenceSearchNode,
    CreateConfluencePageNode,
    DecideConfluenceDestinationNode,
    DocumentQANode,
    EnsureAnalysisContextNode,
    GetConfluencePageNode,
    KnowledgeEnrichmentNode,
    SelectConfluencePageNode,
    StoreConfluencePageInfoNode,
    SummarizeFromConfluenceNode,
    ToolsNode,
    UnderstandConfluenceQueryNode,
    VerifierNode,
)
from .nodes.utils import is_tool_name
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
    code_generation_retry_count: NotRequired[
        int
    ]  # Track code generation retries after errors
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
    - VERIFIER_NODE prefix (optional) for the verifier agent

    For example:
        CHAT_NODE__llm_model_provider=openai
        CHAT_NODE__llm_model_name=gpt-4o
        CHAT_NODE__temperature=0.1
        CHAT_NODE__api_key=sk-...

        CODING_NODE__llm_model_provider=openai
        CODING_NODE__llm_model_name=gpt-4o
        CODING_NODE__temperature=0.1
        CODING_NODE__api_key=sk-...

        VERIFIER_NODE__llm_model_provider=openai
        VERIFIER_NODE__llm_model_name=gpt-4o
        VERIFIER_NODE__temperature=0.0
        VERIFIER_NODE__api_key=sk-...

    If CODING_NODE is not configured, the main LLM (CHAT_NODE) will be used for code generation.
    If VERIFIER_NODE is not configured, the main LLM (CHAT_NODE) with adjusted temperature (0.0-0.2) will be used for verification.

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

    # Filter tools for main agent - exclude run_analysis (only available to coding agent)
    main_agent_tools = [
        tool for tool in tools if not is_tool_name(tool.name, "run_analysis")
    ]
    main_agent_tool_names = [tool.name for tool in main_agent_tools]
    logger.info("=" * 60)
    logger.info(
        "Main agent tools (run_analysis excluded): %d tools", len(main_agent_tools)
    )
    for tool_name in main_agent_tool_names:
        logger.info("  - %s", tool_name)
    logger.info("=" * 60)

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
        llm_with_tools = llm.bind_tools(main_agent_tools, tool_choice=tool_choice)
    else:
        llm_with_tools = llm.bind_tools(main_agent_tools)

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

    # Initialize the verifier LLM if configured, otherwise use llm_json (low-temperature main LLM)
    if settings.verifier_llm:
        logger.info(
            f"Using separate verifier LLM: {settings.verifier_llm.llm_model_name} "
            f"(provider: {settings.verifier_llm.llm_model_provider}, "
            f"temperature: {settings.verifier_llm.temperature})"
        )
        verifier_llm = initialize_llm(settings.verifier_llm)
    else:
        logger.info(
            "No separate verifier LLM configured, using main LLM with adjusted temperature (0.0-0.2) for verification"
        )
        verifier_llm = llm_json

    # All old node function definitions have been moved to node classes.
    # Node instances are created below (before graph construction).
    # Create node instances
    classify_query_node = ClassifyQueryNode(llm_json)
    document_qa_node = DocumentQANode(llm_with_tools)
    knowledge_enrichment_node = KnowledgeEnrichmentNode(
        llm, main_agent_tools, tool_choice
    )
    agent_node = AgentNode(llm_with_tools)
    code_generation_node = CodeGenerationNode(
        coding_llm_with_tools, coding_tools, coding_tool_choice
    )
    tools_node = ToolsNode(tools)
    verifier_node = VerifierNode(verifier_llm)

    # Confluence Export nodes
    ensure_analysis_context_node = EnsureAnalysisContextNode()
    build_confluence_page_draft_node = BuildConfluencePageDraftNode(llm)
    decide_confluence_destination_node = DecideConfluenceDestinationNode(
        settings.confluence_space_key_analytics
    )
    create_confluence_page_node = CreateConfluencePageNode(
        tools, settings.confluence_space_key_analytics
    )
    store_confluence_page_info_node = StoreConfluencePageInfoNode()

    # Confluence Read nodes
    understand_confluence_query_node = UnderstandConfluenceQueryNode(llm_json)
    confluence_search_node = ConfluenceSearchNode(tools)
    select_confluence_page_node = SelectConfluencePageNode(llm_json)
    get_confluence_page_node = GetConfluencePageNode(tools)
    summarize_from_confluence_node = SummarizeFromConfluenceNode(llm)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify", classify_query_node)
    workflow.add_node("document_qa", document_qa_node)
    workflow.add_node("knowledge_enrichment", knowledge_enrichment_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("verifier", verifier_node)

    # Confluence Export subflow nodes
    workflow.add_node("ensure_analysis_context", ensure_analysis_context_node)
    workflow.add_node("build_confluence_page_draft", build_confluence_page_draft_node)
    workflow.add_node(
        "decide_confluence_destination", decide_confluence_destination_node
    )
    workflow.add_node("create_confluence_page", create_confluence_page_node)
    workflow.add_node("store_confluence_page_info", store_confluence_page_info_node)

    # Confluence Read subflow nodes
    workflow.add_node("understand_confluence_query", understand_confluence_query_node)
    workflow.add_node("confluence_search", confluence_search_node)
    workflow.add_node("select_confluence_page", select_confluence_page_node)
    workflow.add_node("get_confluence_page", get_confluence_page_node)
    workflow.add_node("summarize_from_confluence", summarize_from_confluence_node)

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
            if is_tool_name(last_tool_message.name, "get_dataset_schema"):
                # get_dataset_schema is only called by main agent now
                # Route back to agent to continue gathering information or route to code generation
                logger.info(
                    "get_dataset_schema called by main agent, routing back to agent"
                )
                return "agent"

            # If run_analysis succeeded, reset retry count
            if is_tool_name(last_tool_message.name, "run_analysis"):
                if hasattr(last_tool_message, "content"):
                    content = last_tool_message.content
                    if isinstance(content, str):
                        import json

                        try:
                            result_data = json.loads(content)
                            if (
                                isinstance(result_data, dict)
                                and result_data.get("success")
                                and not result_data.get("error")
                            ):
                                # Success - reset retry count
                                logger.info(
                                    "run_analysis succeeded, resetting code generation retry count"
                                )
                                # Note: We can't update state here, but the agent node will handle it
                                # The retry count will naturally reset on next error
                        except (json.JSONDecodeError, TypeError):
                            pass

        # If run_analysis failed and we have error, route to agent for error analysis and retry
        if last_tool_message and hasattr(last_tool_message, "name"):
            if is_tool_name(last_tool_message.name, "run_analysis"):
                # Check if there's an error in the result
                if hasattr(last_tool_message, "content"):
                    content = last_tool_message.content
                    if isinstance(content, str):
                        # Check for error indicators
                        import json

                        has_error = False
                        error_message = None

                        # Try to parse as JSON to extract error
                        try:
                            if isinstance(content, str):
                                result_data = json.loads(content)
                                if isinstance(result_data, dict):
                                    if result_data.get("error") or not result_data.get(
                                        "success", True
                                    ):
                                        has_error = True
                                        error_message = result_data.get(
                                            "error", "Unknown error"
                                        )
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, check for error indicators in string
                            if (
                                '"error"' in content.lower()
                                or '"success": false' in content.lower()
                                or "error" in content.lower()
                            ):
                                has_error = True
                                error_message = content

                        if has_error:
                            # Route back to agent to analyze the error and retry code generation
                            logger.info(
                                "run_analysis failed with error, routing back to agent for error analysis and retry"
                            )
                            if error_message:
                                logger.info(f"Error message: {error_message[:200]}...")
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
