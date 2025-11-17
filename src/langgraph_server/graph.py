"""LangGraph definition for the data analysis agent."""

import logging
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import NotRequired, TypedDict

from .mcp_tool_loader import MCPToolLoader
from .prompts import (
    ANALYSIS_PROMPT,
    CLASSIFICATION_PROMPT,
    DOCUMENT_QA_PROMPT,
    KNOWLEDGE_ENRICHMENT_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


class AgentState(TypedDict):
    """State for the data analysis agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: NotRequired[int]
    query_classification: NotRequired[Literal["DOCUMENT_QA", "DATA_ANALYSIS", "BOTH"]]
    knowledge_context: NotRequired[str]


async def create_agent(model_name: str = "gpt-5-mini", temperature: float = 0.1):
    """
    Create the data analysis agent graph.

    Args:
        model_name: Name of the LLM model to use
        temperature: Temperature for the LLM

    Returns:
        Compiled LangGraph StateGraph
    """
    logger.info(f"Creating agent with model: {model_name}")

    # Load MCP tools
    mcp_tool_loader = MCPToolLoader()
    # Load tools once and keep them for the agent lifetime
    tools = await mcp_tool_loader._load_all_servers()

    # Initialize the LLM
    llm = init_chat_model(
        model=model_name,
        model_provider="openai",
        temperature=temperature,
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create classifier node
    def classify_query(state: AgentState):
        """Classify the user query into DOCUMENT_QA, DATA_ANALYSIS, or BOTH."""
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
        else:
            # Use LLM to classify
            prompt = CLASSIFICATION_PROMPT.invoke({"messages": [user_message]})
            response = llm.invoke(prompt.messages)
            classification_text = response.content.strip().upper()

            # Parse classification
            if "DOCUMENT_QA" in classification_text:
                classification = "DOCUMENT_QA"
            elif "BOTH" in classification_text:
                classification = "BOTH"
            else:
                classification = "DATA_ANALYSIS"

        logger.info(f"Query classified as: {classification}")
        logger.info("=" * 60)
        return {"query_classification": classification}

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

    # Set entry point
    workflow.set_entry_point("classify")

    # Route from classifier
    def route_after_classify(state: AgentState) -> str:
        """Route based on query classification."""
        classification = state.get("query_classification", "DATA_ANALYSIS")
        logger.info(f"Routing based on classification: {classification}")

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
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

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
