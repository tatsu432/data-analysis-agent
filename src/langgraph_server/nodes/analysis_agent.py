"""Analysis agent node with strict tool masking."""

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..prompts.analysis_agent import ANALYSIS_AGENT_PROMPT
from .base import BaseNode
from .utils import is_tool_name

logger = logging.getLogger(__name__)


class AnalysisAgentNode(BaseNode):
    """Analysis agent that only sees analysis tools."""

    def __init__(self, llm_with_tools: BaseChatModel):
        """Initialize the analysis agent node.

        Args:
            llm_with_tools: LLM with analysis tools bound (only list_datasets, get_dataset_schema)
        """
        super().__init__("analysis_agent")
        self.llm_with_tools = llm_with_tools

    def __call__(self, state: dict) -> dict:
        """Execute analysis agent reasoning."""
        self.log_node_start()
        messages = state["messages"]

        prompt = ANALYSIS_AGENT_PROMPT.invoke({"messages": messages})
        logger.info("Invoking analysis agent LLM...")
        response = self.llm_with_tools.invoke(prompt.messages)

        # Check if we have schema information and should route to code generation
        # Look for get_dataset_schema calls in the conversation
        has_schema_info = False
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if is_tool_name(
                        getattr(tool_call, "name", ""), "get_dataset_schema"
                    ):
                        has_schema_info = True
                        break
                if has_schema_info:
                    break

        # Check if code generation is needed
        response_content = getattr(response, "content", "")

        # CRITICAL: Detect if AnalysisAgent incorrectly generated Python code
        # AnalysisAgent should NEVER generate code - it should only output CODE_GENERATION_NEEDED
        contains_python_code = False
        if isinstance(response_content, str):
            # Check for Python code patterns
            python_indicators = [
                r"```python",  # Python code block
                r"```\s*\n\s*import\s+",  # Code block with import
                r"import\s+(pandas|numpy|matplotlib|pd|np|plt)",  # Direct imports
                r"df\s*=",  # DataFrame assignment
                r"result_df\s*=",  # Result assignment
                r"plt\.(figure|plot|savefig)",  # Plotting calls
                r"pd\.(read_csv|DataFrame|merge)",  # Pandas operations
            ]
            for pattern in python_indicators:
                if re.search(pattern, response_content, re.IGNORECASE | re.MULTILINE):
                    contains_python_code = True
                    logger.warning(
                        f"AnalysisAgent incorrectly generated Python code! "
                        f"Detected pattern: {pattern}. Forcing routing to code_agent."
                    )
                    break

        # Route to code generation if:
        # 1. Response contains "CODE_GENERATION_NEEDED"
        # 2. Response contains Python code (AnalysisAgent should never do this)
        # 3. We have schema info and no more tool calls needed
        needs_code_gen = (
            (
                isinstance(response_content, str)
                and "CODE_GENERATION_NEEDED" in response_content
            )
            or contains_python_code
            or (
                has_schema_info
                and not (hasattr(response, "tool_calls") and response.tool_calls)
            )
        )

        if needs_code_gen:
            if contains_python_code:
                logger.warning(
                    "AnalysisAgent generated code instead of routing. "
                    "This code will be passed to CodeAgent for proper generation."
                )
            logger.info("Analysis agent indicates code generation is needed")
            routing_msg = AIMessage(content="CODE_GENERATION_NEEDED")
            self.log_node_end()
            return {"messages": [routing_msg]}

        # Log tool calls if any
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(
                f"Analysis agent requested {len(response.tool_calls)} tool call(s):"
            )
            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                logger.info(f"  - {tool_name}")
        else:
            logger.info("Analysis agent response (no tool calls)")

        self.log_node_end()
        return {"messages": [response]}
