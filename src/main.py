"""Main entry point for the data analysis agent."""

import asyncio
import base64
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.graph import create_agent

# Suppress LangSmith UUID warning
warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7 for run and trace identifiers.*",
    category=UserWarning,
    module="pydantic.v1.main",
)

# Load environment variables
load_dotenv()


async def run_analysis(query: str, thread_id: str = "default-thread"):
    """
    Run a data analysis query through the agent.

    Args:
        query: Natural language query about the data
        thread_id: Thread ID for conversation memory
    """
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    # Create the agent
    app = create_agent()

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
    final_state = None
    plot_saved = False
    plot_path = None

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
                        # Check for tool results with plots (ToolMessage)
                        if isinstance(message, ToolMessage):
                            try:
                                # Handle both string and dict content
                                if isinstance(message.content, str):
                                    tool_result = json.loads(message.content)
                                elif isinstance(message.content, dict):
                                    tool_result = message.content
                                else:
                                    continue

                                # Prefer plot_path if available (more efficient)
                                if (
                                    "plot_path" in tool_result
                                    and tool_result["plot_path"]
                                ):
                                    plot_path = Path(tool_result["plot_path"])
                                    if plot_path.exists():
                                        plot_saved = True
                                        print(f"\n📊 Plot saved to: {plot_path}\n")
                                # Fallback to plot_base64 if plot_path not available
                                elif (
                                    "plot_base64" in tool_result
                                    and tool_result["plot_base64"]
                                ):
                                    # Save the plot to a file in img folder
                                    project_root = Path(__file__).parent
                                    img_dir = project_root / "img"
                                    img_dir.mkdir(parents=True, exist_ok=True)
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    plot_path = img_dir / f"plot_{timestamp}.png"

                                    plot_bytes = base64.b64decode(
                                        tool_result["plot_base64"]
                                    )
                                    with open(plot_path, "wb") as f:
                                        f.write(plot_bytes)

                                    plot_saved = True
                                    print(f"\n📊 Plot saved to: {plot_path}\n")
                            except (
                                json.JSONDecodeError,
                                TypeError,
                                KeyError,
                                ValueError,
                            ):
                                # Silently ignore parsing errors
                                pass

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

        final_state = event

    # Get final response from the final state
    if final_state and isinstance(final_state, dict):
        # Get the last message from the agent node
        if "agent" in final_state and isinstance(final_state["agent"], dict):
            if "messages" in final_state["agent"]:
                messages_list = final_state["agent"]["messages"]
                if messages_list:
                    last_message = messages_list[-1]
                    if hasattr(last_message, "content") and last_message.content:
                        print(f"\n{'=' * 60}")
                        print("Final Answer:")
                        print(f"{'=' * 60}")
                        print(last_message.content)
                        print(f"{'=' * 60}\n")

                        if plot_saved and plot_path:
                            print(
                                f"💡 Tip: You can view the plot by opening: {plot_path}\n"
                            )


async def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("Data Analysis Agent - Interactive Mode")
    print("=" * 60)
    print("Enter your queries about the COVID-19 data.")
    print("Type 'exit' or 'quit' to end the session.\n")

    app = create_agent()
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
            final_content = None

            plot_saved = False
            plot_path = None

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
                                # Check for tool results with plots (ToolMessage)
                                if isinstance(message, ToolMessage):
                                    try:
                                        if isinstance(message.content, str):
                                            tool_result = json.loads(message.content)
                                        elif isinstance(message.content, dict):
                                            tool_result = message.content
                                        else:
                                            continue

                                        # Prefer plot_path if available (more efficient)
                                        if (
                                            "plot_path" in tool_result
                                            and tool_result["plot_path"]
                                        ):
                                            plot_path = Path(tool_result["plot_path"])
                                            if plot_path.exists():
                                                plot_saved = True
                                                print(
                                                    f"\n📊 Plot saved to: {plot_path}\n"
                                                )
                                        # Fallback to plot_base64 if plot_path not available
                                        elif (
                                            "plot_base64" in tool_result
                                            and tool_result["plot_base64"]
                                        ):
                                            # Save the plot to a file in img folder
                                            project_root = Path(__file__).parent
                                            img_dir = project_root / "img"
                                            img_dir.mkdir(parents=True, exist_ok=True)
                                            timestamp = datetime.now().strftime(
                                                "%Y%m%d_%H%M%S"
                                            )
                                            plot_path = (
                                                img_dir / f"plot_{timestamp}.png"
                                            )

                                            plot_bytes = base64.b64decode(
                                                tool_result["plot_base64"]
                                            )
                                            with open(plot_path, "wb") as f:
                                                f.write(plot_bytes)

                                            plot_saved = True
                                            print(f"\n📊 Plot saved to: {plot_path}\n")
                                    except (
                                        json.JSONDecodeError,
                                        TypeError,
                                        KeyError,
                                        ValueError,
                                    ):
                                        pass

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
                                        final_content = content

            if final_content:
                print(f"\n{'─' * 60}\n")

            if plot_saved and plot_path:
                print(f"💡 Tip: You can view the plot by opening: {plot_path}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Non-interactive mode: run a single query
        query = " ".join(sys.argv[1:])
        asyncio.run(run_analysis(query))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
