"""Streamlit UI for the data analysis agent."""

import asyncio
import base64
import json
import sys
import warnings
from io import BytesIO
from pathlib import Path

# Suppress LangSmith UUID warning
warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7 for run and trace identifiers.*",
    category=UserWarning,
    module="pydantic.v1.main",
)

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import project modules
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from agent.graph import create_agent  # noqa: E402

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analysis Agent")
st.markdown("Ask questions about COVID-19 data for Japanese prefectures")

# Initialize session state
if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        st.session_state.agent = create_agent()
        st.session_state.thread_id = "streamlit-session"
        st.session_state.messages = []
        st.session_state.debug_info = []  # Store debug/technical information

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display plot if available
        if "plot" in message:
            st.image(message["plot"], caption="Analysis Plot")

# Debug information expander (only show if there's debug info)
if st.session_state.debug_info:
    with st.expander("🔧 Debug Information (for engineers)", expanded=False):
        for idx, debug_entry in enumerate(st.session_state.debug_info):
            st.subheader(f"Query: {debug_entry['query']}")
            for debug_item in debug_entry["debug_data"]:
                st.markdown(f"**Tool: {debug_item.get('tool', 'unknown')}**")
                if debug_item["type"] == "tool_result":
                    # Pretty print JSON data
                    st.json(debug_item["data"])
                else:
                    st.text(debug_item["data"])
                st.markdown("---")
            if idx < len(st.session_state.debug_info) - 1:
                st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about the data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize response variables
    full_response = ""
    plot_base64 = None
    current_debug_info = []  # Debug info for this response

    # Get agent response
    with st.chat_message("assistant"):
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
            },
            "recursion_limit": 50,
        }

        messages = [HumanMessage(content=prompt)]

        # Collect response
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        response_data = {"full_response": "", "plot_base64": None}

        # Status messages for different nodes
        status_messages = {
            "agent": "🤔 Thinking about the analysis plan...",
            "tools": "⚙️ Executing analysis...",
        }

        def get_tool_status_message(tool_name: str) -> str:
            """Get a specific status message based on the tool being called."""
            tool_status_map = {
                "get_dataset_schema_tool": "📊 Analyzing dataset structure...",
                "run_covid_analysis_tool": "⚙️ Executing data analysis code...",
            }
            return tool_status_map.get(tool_name, "⚙️ Running tool...")

        async def stream_response():
            current_node = None
            async for event in st.session_state.agent.astream(
                {"messages": messages},
                config=config,
                stream_mode="updates",
            ):
                # Handle different event structures
                if isinstance(event, dict):
                    for node_name, node_output in event.items():
                        # Update status based on current node
                        if node_name != current_node:
                            current_node = node_name
                            if node_name in status_messages:
                                status_placeholder.info(status_messages[node_name])

                        if isinstance(node_output, dict) and "messages" in node_output:
                            for message in node_output["messages"]:
                                # Handle ToolMessage (debug/technical info)
                                if isinstance(message, ToolMessage):
                                    # Update status for specific tool
                                    tool_name = getattr(message, "name", "unknown")
                                    status_placeholder.info(
                                        get_tool_status_message(tool_name)
                                    )

                                    # Store debug info but don't show to user
                                    try:
                                        if isinstance(message.content, str):
                                            # Try to parse as JSON
                                            try:
                                                tool_result = json.loads(
                                                    message.content
                                                )
                                                # Store debug info
                                                debug_entry = {
                                                    "type": "tool_result",
                                                    "tool": tool_name,
                                                    "data": tool_result,
                                                }
                                                current_debug_info.append(debug_entry)

                                                # Extract plot if available
                                                if "plot_base64" in tool_result:
                                                    response_data["plot_base64"] = (
                                                        tool_result["plot_base64"]
                                                    )
                                            except json.JSONDecodeError:
                                                # Not JSON, store as text
                                                debug_entry = {
                                                    "type": "tool_output",
                                                    "tool": tool_name,
                                                    "data": message.content,
                                                }
                                                current_debug_info.append(debug_entry)
                                    except Exception:
                                        pass

                                # Handle AIMessage (user-facing content)
                                elif isinstance(message, AIMessage):
                                    # Check if AI is about to call tools
                                    has_tool_calls = (
                                        hasattr(message, "tool_calls")
                                        and message.tool_calls
                                        and len(message.tool_calls) > 0
                                    )

                                    if has_tool_calls:
                                        status_placeholder.info(
                                            "🔧 Preparing to execute tools..."
                                        )

                                    # Show content only if it's not just tool calls
                                    if hasattr(message, "content") and message.content:
                                        content = str(message.content)
                                        # Filter out content that looks like raw tool output
                                        # (large JSON blocks that are likely debug info)
                                        if content and content.strip():
                                            content_stripped = content.strip()
                                            # Check if content looks like raw tool output
                                            # (JSON with tool-specific keys like "columns", "dtypes", "stdout", etc.)
                                            is_raw_tool_output = False
                                            if (
                                                content_stripped.startswith("{")
                                                and content_stripped.endswith("}")
                                                and len(content_stripped) > 500
                                            ):
                                                try:
                                                    parsed = json.loads(
                                                        content_stripped
                                                    )
                                                    # Check for tool output indicators
                                                    tool_output_keys = [
                                                        "columns",
                                                        "dtypes",
                                                        "sample_rows",
                                                        "row_count",
                                                        "stdout",
                                                        "error",
                                                        "result_df_preview",
                                                        "result_df_row_count",
                                                    ]
                                                    if any(
                                                        key in parsed
                                                        for key in tool_output_keys
                                                    ):
                                                        is_raw_tool_output = True
                                                except json.JSONDecodeError:
                                                    pass

                                            if not is_raw_tool_output:
                                                response_data["full_response"] += (
                                                    content + "\n\n"
                                                )
                                                response_placeholder.markdown(
                                                    response_data["full_response"]
                                                )
                                                # Clear status when we get AI response
                                                status_placeholder.empty()

            # Clear status at the end
            status_placeholder.empty()

        # Run async function
        asyncio.run(stream_response())

        # Extract final values from response_data
        full_response = response_data["full_response"].strip()
        plot_base64 = response_data["plot_base64"]

        # Display final response
        if full_response:
            response_placeholder.markdown(full_response)
        else:
            response_placeholder.markdown(
                "I'm working on your request. Please check the debug information below for details."
            )

        # Display plot if available
        if plot_base64:
            plot_bytes = base64.b64decode(plot_base64)
            st.image(plot_bytes, caption="Analysis Plot")

        # Store debug info in session state
        if current_debug_info:
            st.session_state.debug_info.append(
                {
                    "query": prompt,
                    "debug_data": current_debug_info,
                }
            )

    # Add assistant response to history (only user-facing content)
    message_to_add = {"role": "assistant", "content": full_response}
    if plot_base64:
        plot_bytes = base64.b64decode(plot_base64)
        message_to_add["plot"] = BytesIO(plot_bytes)
    st.session_state.messages.append(message_to_add)

# Sidebar with example queries
with st.sidebar:
    st.header("Example Queries")
    example_queries = [
        "How does the number of patients vary from January to July 2025 in Tokyo?",
        "Generate a line plot of the number of patients from May to August 2024 in each prefecture of the Kanto region.",
        "What characteristics does the patient count data have overall?",
        "Show me the top 5 prefectures with the highest total cases in 2024.",
        "Compare the patient counts between Tokyo and Osaka over time.",
    ]

    for query in example_queries:
        if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.debug_info = []
        st.rerun()

    # Show debug info count
    if st.session_state.debug_info:
        st.markdown("---")
        st.info(f"🔧 {len(st.session_state.debug_info)} debug session(s) available")
        if st.button("Clear Debug Info"):
            st.session_state.debug_info = []
            st.rerun()
