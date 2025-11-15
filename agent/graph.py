"""LangGraph definition for the data analysis agent."""

import logging
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import NotRequired

from agent.prompts import ANALYSIS_PROMPT
from agent.tools import get_dataset_schema, run_covid_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the data analysis agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: NotRequired[int]


# Convert functions to LangChain tools
@tool
def get_dataset_schema_tool() -> dict:
    """Get schema information about the COVID-19 dataset including columns, data types, and sample rows."""
    logger.info("Tool called: get_dataset_schema_tool")
    try:
        result = get_dataset_schema()
        logger.info("Tool completed: get_dataset_schema_tool")
        return result
    except Exception as e:
        logger.error(
            f"Tool failed: get_dataset_schema_tool - {type(e).__name__}: {str(e)}"
        )
        raise


@tool
def run_covid_analysis_tool(code: str) -> dict:
    """
    Execute Python code for COVID-19 data analysis.

    Args:
        code: Python code string to execute. The dataset is available as `df`.
              Date columns are automatically converted to datetime.
              Use pandas, numpy, and matplotlib.pyplot. Assign results to `result_df`
              and save plots as `analysis_plot.png`.

    Returns:
        Dictionary with execution results including:
        - stdout: captured standard output
        - error: error or warning messages (check this!)
        - result_df_preview: preview of result_df (first 10 rows)
        - result_df_row_count: number of rows in result_df (check if 0!)
        - plot_base64: base64-encoded plot if created
        - plot_valid: boolean indicating if plot contains data (check this!)
        - plot_validation_message: message about plot validation
        - success: boolean indicating if execution succeeded

        IMPORTANT: Always check result_df_row_count and plot_valid before interpreting results.
        If result_df_row_count is 0 or plot_valid is False, your query returned no data.
    """
    logger.info("Tool called: run_covid_analysis_tool")
    logger.info(f"Tool input - code length: {len(code)} characters")
    try:
        result = run_covid_analysis(code)
        logger.info("Tool completed: run_covid_analysis_tool")
        return result
    except Exception as e:
        logger.error(
            f"Tool failed: run_covid_analysis_tool - {type(e).__name__}: {str(e)}"
        )
        raise


def create_agent(model_name: str = "gpt-5-mini", temperature: float = 0.1):
    """
    Create the data analysis agent graph.

    Args:
        model_name: Name of the LLM model to use
        temperature: Temperature for the LLM

    Returns:
        Compiled LangGraph StateGraph
    """
    logger.info(f"Creating agent with model: {model_name}")

    # Initialize the LLM
    llm = init_chat_model(
        model=model_name,
        model_provider="openai",
        temperature=temperature,
    )

    # Bind tools to the LLM
    tools = [get_dataset_schema_tool, run_covid_analysis_tool]
    llm_with_tools = llm.bind_tools(tools)

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
        prompt = ANALYSIS_PROMPT.invoke({"messages": messages})
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

    # Create a custom tool node wrapper for logging
    def call_tools(state: AgentState):
        """Call tools with logging."""
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

        # Use ToolNode to execute tools
        tool_node = ToolNode(tools)
        result = tool_node.invoke(state)

        # Log tool results
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "name"):
                    tool_name = getattr(msg, "name", "unknown")
                    logger.info(f"Tool result received from: {tool_name}")

        return result

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("Agent graph created successfully")
    return app


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine whether to continue or end based on the last message."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # Otherwise, end
    return "end"
