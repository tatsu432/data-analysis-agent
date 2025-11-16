"""Entry point for running the LangGraph server as a module.

Usage:
    python -m src.langgraph_server
"""

import asyncio
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress LangSmith UUID warning
warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7 for run and trace identifiers.*",
    category=UserWarning,
    module="pydantic.v1.main",
)

from langchain_core.messages import HumanMessage

from src.langgraph_server.graph import create_agent

# Load environment variables
load_dotenv()


async def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("Data Analysis Agent - Interactive Mode")
    print("=" * 60)
    print("Enter your queries about the COVID-19 data.")
    print("Type 'exit' or 'quit' to end the session.\n")

    app = await create_agent()
    thread_id = "interactive-session"

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 50,
    }

    while True:
        try:
            query = input("\nYour query: ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            messages = [HumanMessage(content=query)]

            print("\nAgent is analyzing...\n")

            async for event in app.astream(
                {"messages": messages},
                config=config,
                stream_mode="updates",
            ):
                # Handle different event structures
                if isinstance(event, dict):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for message in node_output["messages"]:
                                # Print tool calls
                                if (
                                    hasattr(message, "tool_calls")
                                    and message.tool_calls
                                ):
                                    for tool_call in message.tool_calls:
                                        tool_name = (
                                            tool_call.get("name", "unknown")
                                            if isinstance(tool_call, dict)
                                            else getattr(tool_call, "name", "unknown")
                                        )
                                        print(f"[Calling tool: {tool_name}]\n")

                                # Print AI responses
                                if hasattr(message, "content") and message.content:
                                    content = str(message.content)
                                    if content and content.strip():
                                        print(f"Agent: {content}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


async def run_single_query(query: str, thread_id: str = "default-thread"):
    """Run a single query through the agent."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    # Create the agent
    app = await create_agent()

    # Configure thread for memory
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 50,
    }

    # Create initial message
    messages = [HumanMessage(content=query)]

    # Stream the agent's response
    print("Agent is analyzing...\n")

    async for event in app.astream(
        {"messages": messages},
        config=config,
        stream_mode="updates",
    ):
        # Handle different event structures
        if isinstance(event, dict):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict) and "messages" in node_output:
                    for message in node_output["messages"]:
                        # Print tool calls
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                tool_name = (
                                    tool_call.get("name", "unknown")
                                    if isinstance(tool_call, dict)
                                    else getattr(tool_call, "name", "unknown")
                                )
                                print(f"[Calling tool: {tool_name}]\n")

                        # Print AI responses
                        if hasattr(message, "content") and message.content:
                            content = str(message.content)
                            if content and content.strip():
                                print(f"Agent: {content}\n")


def main():
    """Main entry point for LangGraph server."""
    if len(sys.argv) > 1:
        # Non-interactive mode: run a single query
        query = " ".join(sys.argv[1:])
        asyncio.run(run_single_query(query))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()

