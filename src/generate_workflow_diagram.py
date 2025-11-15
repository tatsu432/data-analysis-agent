"""Script to generate workflow diagram from the agent graph.

This script automatically generates a visual representation of the agent workflow.
Run this script whenever you modify the workflow to update the diagram.
"""

import sys
from pathlib import Path

# Add src directory to Python path (must be before other imports)
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agent.graph import create_agent, generate_workflow_diagram

if __name__ == "__main__":
    print("Creating agent...")
    app = create_agent()

    print("Generating workflow diagram...")
    # Generate PNG diagram
    png_path = generate_workflow_diagram(app, "agent_workflow.png", format="png")
    print(f"✓ PNG diagram saved to: {png_path}")

    # Generate Mermaid source code
    mmd_path = generate_workflow_diagram(app, "agent_workflow.mmd", format="mermaid")
    print(f"✓ Mermaid source saved to: {mmd_path}")

    print(
        "\nDone! The workflow diagram has been generated from the current graph structure."
    )
