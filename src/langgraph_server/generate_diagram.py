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
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.langgraph_server.graph import create_agent

logger = logging.getLogger(__name__)


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
        >>> from src.langgraph_server.graph import create_agent
        >>> from src.langgraph_server.generate_diagram import generate_workflow_diagram
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
