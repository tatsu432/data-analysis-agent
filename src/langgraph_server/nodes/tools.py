"""Tools node for executing tool calls."""

import json
import logging

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from .base import BaseNode

logger = logging.getLogger(__name__)


class ToolsNode(BaseNode):
    """Call tools with logging (async to support MCP tools)."""

    def __init__(self, tools: list[BaseTool]):
        """Initialize the tools node.

        Args:
            tools: List of available tools
        """
        super().__init__("tools")
        self.tools = tools
        self.tool_node = ToolNode(tools)

    async def __call__(self, state: dict) -> dict:
        """Execute tool calls."""
        self.log_node_start()
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if (
            last_message
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            logger.info(f"Executing {len(last_message.tool_calls)} tool call(s):")
            for tool_call in last_message.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                tool_args = getattr(tool_call, "args", {})
                logger.info(f"  - Tool: {tool_name}")
                logger.debug(f"    Args: {tool_args}")

        # Use ToolNode to execute tools (async for MCP tools)
        result = await self.tool_node.ainvoke(state)

        # Log tool results and detect plot generation
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "name"):
                    tool_name = getattr(msg, "name", "unknown")
                    logger.info(f"Tool result received from: {tool_name}")

                    # Detect plot generation from run_analysis results
                    if tool_name == "run_analysis" and hasattr(msg, "content"):
                        try:
                            content = msg.content
                            if isinstance(content, str):
                                # Try to parse as JSON
                                try:
                                    result_data = json.loads(content)
                                except (json.JSONDecodeError, TypeError):
                                    result_data = None

                                if result_data and isinstance(result_data, dict):
                                    plot_path = result_data.get("plot_path")
                                    plot_valid = result_data.get("plot_valid")

                                    if plot_path:
                                        logger.info("=" * 60)
                                        logger.info("PLOT DETECTED:")
                                        logger.info(f"  Plot path: {plot_path}")
                                        logger.info(f"  Plot valid: {plot_valid}")
                                        if plot_valid:
                                            logger.info(
                                                "  ✅ Plot was successfully generated and validated"
                                            )
                                        else:
                                            validation_msg = result_data.get(
                                                "plot_validation_message", "Unknown"
                                            )
                                            logger.warning(
                                                f"  ⚠️  Plot was generated but validation failed: {validation_msg}"
                                            )
                                        logger.info("=" * 60)
                        except Exception as e:
                            logger.debug(f"Error checking for plot: {e}")

        self.log_node_end()
        return result
