"""Main entry point for the data analysis agent.

This is a convenience wrapper around `python -m src.langgraph_server`.
The actual entry point is in src/langgraph_server/__main__.py
"""

import subprocess
import sys


if __name__ == "__main__":
    # Run the langgraph_server module as a script
    subprocess.run([sys.executable, "-m", "src.langgraph_server"] + sys.argv[1:])
