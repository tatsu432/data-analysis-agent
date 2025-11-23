"""Streamlit UI for the data analysis agent.

This UI connects to the LangGraph Server via HTTP API.
"""

import json
import os
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuration
LANGGRAPH_SERVER_URL = os.getenv("LANGGRAPH_SERVER_URL", "http://localhost:2024")
GRAPH_NAME = os.getenv("LANGGRAPH_GRAPH_NAME", "data_analysis_agent")
ASSISTANT_ID = os.getenv(
    "LANGGRAPH_ASSISTANT_ID", "c0cc005a-576b-4f63-9375-bf8ac46e15c7"
)

# Page config
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analysis Agent")
st.markdown("Ask questions about COVID-19 data, Confluence pages, and other data sources (PDF, Excel, etc.)")


def get_or_create_assistant():
    """Get or create the assistant_id for a graph."""
    if ASSISTANT_ID:
        return ASSISTANT_ID

    try:
        response = requests.get(
            f"{LANGGRAPH_SERVER_URL}/assistants/{GRAPH_NAME}",
            timeout=5,
        )
        if response.status_code == 200:
            assistant = response.json()
            return assistant.get("assistant_id") or GRAPH_NAME
    except Exception:
        pass

    try:
        response = requests.post(
            f"{LANGGRAPH_SERVER_URL}/assistants",
            json={"graph_id": GRAPH_NAME},
            timeout=5,
        )
        if response.status_code in [200, 201]:
            assistant = response.json()
            return assistant.get("assistant_id") or GRAPH_NAME
    except Exception:
        pass

    return GRAPH_NAME


@st.cache_resource
def check_server_health():
    """Check if the LangGraph server is running."""
    try:
        response = requests.get(f"{LANGGRAPH_SERVER_URL}/docs", timeout=5)
        return response.status_code == 200, None
    except Exception as e:
        return False, str(e)


def get_or_create_thread(thread_id: str):
    """Get or create a thread in LangGraph Server."""
    try:
        response = requests.get(
            f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}",
            timeout=5,
        )
        if response.status_code == 200:
            return thread_id
    except Exception:
        pass

    try:
        response = requests.post(
            f"{LANGGRAPH_SERVER_URL}/threads",
            json={"thread_id": thread_id},
            timeout=5,
        )
        if response.status_code in [200, 201]:
            return thread_id
    except Exception as e:
        raise Exception(f"Failed to create thread: {e}")

    return thread_id


def create_run(thread_id: str, input_data: dict, assistant_id: str):
    """Create a run in LangGraph Server."""
    # Verify assistant exists
    try:
        response = requests.get(
            f"{LANGGRAPH_SERVER_URL}/assistants/{assistant_id}",
            timeout=5,
        )
        if response.status_code != 200:
            raise Exception(f"Assistant {assistant_id} not found")
    except Exception as e:
        raise Exception(f"Error checking assistant: {e}")

    # Create run
    response = requests.post(
        f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}/runs",
        json={"assistant_id": assistant_id, "input": input_data},
        timeout=30,
    )

    if response.status_code in [200, 201]:
        return response.json()
    else:
        error_detail = response.text
        try:
            error_json = response.json()
            error_detail = str(error_json)
        except Exception:
            pass
        raise Exception(
            f"Failed to create run: {response.status_code} - {error_detail}"
        )


def stream_run_events(thread_id: str, run_id: str):
    """Stream events from a run."""
    response = requests.get(
        f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}/runs/{run_id}/stream",
        stream=True,
        timeout=300,
    )
    if response.status_code == 200:
        return response
    else:
        raise Exception(
            f"Failed to stream events: {response.status_code} - {response.text}"
        )


# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check server health
server_healthy, health_info = check_server_health()
if not server_healthy:
    st.error(
        f"❌ Cannot connect to LangGraph Server at {LANGGRAPH_SERVER_URL}\n\n"
        f"Please make sure:\n"
        f"1. The MCP server is running: `python -m src.mcp_server`\n"
        f"2. The LangGraph Server is running: `langgraph dev`\n"
        f"3. The server URL is correct (current: {LANGGRAPH_SERVER_URL})"
    )
    if health_info:
        st.error(f"Error: {health_info}")
    st.stop()

# Get assistant ID
assistant_id = get_or_create_assistant()

# Sidebar
with st.sidebar:
    st.success("✅ Connected to LangGraph Server")
    st.caption(f"Server: {LANGGRAPH_SERVER_URL}")
    st.caption(f"Graph: {GRAPH_NAME}")
    st.caption(f"Assistant ID: {assistant_id}")

    st.markdown("---")
    st.header("Example Queries")
    example_queries = [
        "How does the number of patients vary from January to July 2022 in Tokyo?",
        # "Generate and compare the line plots of the number of patients from January to August 2022 in Tokyo, Chiba, Saitama, Kanagawa.",
        # "What characteristics does the patient count data have overall?",
        # "Can you model the Tokyo's covid case and tell me the model clearly?",
        # "Can you compare the each product's number of patients over the time for GP only?",
        # "Can you generate the line plots of the number of the patients for each product only for those at risk over the time?",
        # "Can you create a regression model where we predict the number of patient for LAGEVRIO by the MR activities? Tell me the fitted model and MAPE."
        "2022年1月から2022年12月までの東京のコロナウイルス感染者数を図にして、要約して"
    ]

    for query in example_queries:
        if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# Display chat history
# Simple rule: if last message is from user, don't show the previous assistant message
for i, message in enumerate(st.session_state.messages):
    # Skip the previous assistant message if we're waiting for a response
    if (
        i == len(st.session_state.messages) - 2
        and message.get("role") == "assistant"
        and st.session_state.messages[-1].get("role") == "user"
    ):
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot" in message:
            st.image(message["plot"], caption="Analysis Plot")


# Chat input
if prompt := st.chat_input("Ask a question about the data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Track when this query started (for plot detection)
    query_start_time = time.time()

    # Get agent response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.info("🤔 Processing your request...")

        try:
            # Ensure thread exists
            thread_id = get_or_create_thread(st.session_state.thread_id)

            # Create run
            run_data = create_run(
                thread_id,
                {"messages": [{"role": "human", "content": prompt}]},
                assistant_id,
            )
            run_id = run_data.get("run_id") or run_data.get("id")

            if not run_id:
                st.error(f"Failed to get run ID from response: {run_data}")
                st.stop()

            # Stream events
            accumulated_content = ""
            current_event_type = None

            for line in stream_run_events(thread_id, run_id).iter_lines():
                if line:
                    line_str = line.decode("utf-8")

                    if line_str.startswith("event: "):
                        current_event_type = line_str[7:].strip()
                    elif line_str.startswith("data: "):
                        data_str = line_str[6:]
                        try:
                            event_data = json.loads(data_str)

                            if current_event_type == "values":
                                if "messages" in event_data:
                                    # Check for tool calls to update status
                                    for msg in event_data["messages"]:
                                        if msg.get("type") == "ai":
                                            tool_calls = msg.get("tool_calls", [])
                                            if tool_calls and len(tool_calls) > 0:
                                                tool_name = tool_calls[0].get(
                                                    "name", "tool"
                                                )
                                                if isinstance(tool_name, dict):
                                                    tool_name = tool_name.get(
                                                        "name", "tool"
                                                    )

                                                if tool_name == "list_datasets":
                                                    status_placeholder.info(
                                                        "📋 Listing available datasets..."
                                                    )
                                                elif tool_name == "get_dataset_schema":
                                                    status_placeholder.info(
                                                        "📊 Examining dataset structure..."
                                                    )
                                                elif tool_name == "run_analysis":
                                                    status_placeholder.info(
                                                        "⚙️ Executing analysis and generating results..."
                                                    )
                                                elif tool_name == "run_covid_analysis":
                                                    status_placeholder.info(
                                                        "⚙️ Executing analysis and generating results..."
                                                    )
                                                else:
                                                    status_placeholder.info(
                                                        f"🔧 Using {tool_name}..."
                                                    )

                                            # Extract AI message content
                                            content = msg.get("content", "")
                                            if content and content.strip():
                                                if content != accumulated_content:
                                                    accumulated_content = content
                                                    response_placeholder.markdown(
                                                        accumulated_content
                                                    )
                                                    status_placeholder.empty()

                            current_event_type = None
                        except json.JSONDecodeError:
                            pass

            full_response = accumulated_content.strip()
            status_placeholder.empty()

            # Check for plot files created after query start
            plot_path = None
            img_dir = project_root / "img"
            if img_dir.exists():
                plot_files = sorted(
                    img_dir.glob("plot_*.png"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for plot_file in plot_files:
                    if plot_file.stat().st_mtime >= query_start_time:
                        plot_path = plot_file
                        st.image(str(plot_path), caption="Analysis Plot")
                        break

        except Exception as e:
            st.error(f"Error: {e}")
            full_response = f"Error: {str(e)}"

        # Display final response
        if full_response:
            response_placeholder.markdown(full_response)
        else:
            response_placeholder.markdown("I'm working on your request...")

    # Add assistant response to history
    message_to_add = {"role": "assistant", "content": full_response}
    if plot_path and plot_path.exists():
        with open(plot_path, "rb") as f:
            message_to_add["plot"] = BytesIO(f.read())
    st.session_state.messages.append(message_to_add)
