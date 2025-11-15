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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the data analysis agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: NotRequired[int]


# Convert functions to LangChain tools
@tool
def get_dataset_schema_tool() -> dict:
    """Get schema information about the COVID-19 dataset including columns, data types, and sample rows."""
    return get_dataset_schema()


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
    return run_covid_analysis(code)


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
        prompt = ANALYSIS_PROMPT.invoke({"messages": messages})
        response = llm_with_tools.invoke(prompt.messages)
        return {"messages": [response]}

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

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
