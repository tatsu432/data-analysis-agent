"""Script to generate workflow diagram from the agent graph.

This script automatically generates a visual representation of the agent workflow.
Run this script whenever you modify the workflow to update the diagram.

Usage:
    python -m src.langgraph_server.generate_diagram
    python -m src.langgraph_server.generate_diagram --format svg
    python -m src.langgraph_server.generate_diagram --output custom_name.png
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.langgraph_server.graph import create_agent, generate_workflow_diagram


def main():
    """Main function to generate workflow diagram."""
    parser = argparse.ArgumentParser(
        description="Generate a visual diagram of the LangGraph agent workflow"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="agent_workflow.png",
        help="Output file path (default: agent_workflow.png)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["png", "svg", "mermaid"],
        default="png",
        help="Output format (default: png)",
    )

    args = parser.parse_args()

    print("Creating agent...")
    app = asyncio.run(create_agent())

    print("Generating workflow diagram...")
    # Generate diagram
    diagram_path = generate_workflow_diagram(app, args.output, format=args.format)
    print(f"✓ Diagram saved to: {diagram_path}")

    print(
        "\nDone! The workflow diagram has been generated from the current graph structure."
    )


if __name__ == "__main__":
    main()
