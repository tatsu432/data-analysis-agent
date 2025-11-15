"""Streamlit UI for the data analysis agent."""
import asyncio
import base64
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import create_agent

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

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display plot if available
        if "plot" in message:
            st.image(message["plot"], caption="Analysis Plot")

# Chat input
if prompt := st.chat_input("Ask a question about the data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id,
                },
                "recursion_limit": 50,
            }
            
            messages = [HumanMessage(content=prompt)]
            
            # Collect response
            response_placeholder = st.empty()
            full_response = ""
            plot_base64 = None
            
            async def stream_response():
                nonlocal full_response, plot_base64
                async for event in st.session_state.agent.astream(
                    {"messages": messages},
                    config=config,
                    stream_mode="updates",
                ):
                    # Handle different event structures
                    if isinstance(event, dict):
                        for node_name, node_output in event.items():
                            if isinstance(node_output, dict) and "messages" in node_output:
                                for message in node_output["messages"]:
                                    # Check for tool results with plots (ToolMessage)
                                    if hasattr(message, "content") and isinstance(message.content, str):
                                        # Try to parse tool result for plots
                                        import json
                                        try:
                                            if "plot_base64" in message.content:
                                                # Try to extract plot from tool result
                                                tool_result = json.loads(message.content)
                                                if "plot_base64" in tool_result:
                                                    plot_base64 = tool_result["plot_base64"]
                                        except (json.JSONDecodeError, TypeError):
                                            pass
                                    
                                    # Display AI responses
                                    if hasattr(message, "content") and message.content:
                                        content = str(message.content)
                                        if content and content.strip():
                                            full_response += content + "\n\n"
                                            response_placeholder.markdown(full_response)
            
            # Run async function
            asyncio.run(stream_response())
            
            # Display final response
            response_placeholder.markdown(full_response)
            
            # Try to extract and display plot from tool results
            # This is a simplified version - in practice, you'd parse tool results more carefully
            if plot_base64:
                plot_bytes = base64.b64decode(plot_base64)
                st.image(plot_bytes, caption="Analysis Plot")
    
    # Add assistant response to history
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
        st.rerun()

