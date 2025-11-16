#!/usr/bin/env python3
"""
Convenience script to run the MCP server.
This is a wrapper around `python -m src.mcp_server`.

For better organization, the actual entry point is in src/mcp_server/__main__.py
"""

import subprocess
import sys

if __name__ == "__main__":
    # Run the module as a script
    subprocess.run([sys.executable, "-m", "src.mcp_server"] + sys.argv[1:])

