"""Code agent node - generates Python code only (no tools)."""

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..prompts.code_agent import CODE_AGENT_PROMPT
from .base import BaseNode
from .utils import extract_content_text

logger = logging.getLogger(__name__)


class CodeAgentNode(BaseNode):
    """Code agent that only generates Python code (no tools)."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the code agent node.

        Args:
            llm: LLM for code generation (no tools)
        """
        super().__init__("code_agent")
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        """Generate Python code for analysis."""
        self.log_node_start()
        messages = state["messages"]

        prompt = CODE_AGENT_PROMPT.invoke({"messages": messages})
        logger.info("Invoking code agent LLM...")
        response = self.llm.invoke(prompt.messages)

        response_content = extract_content_text(getattr(response, "content", None))

        # Extract code from response
        code = None

        # Try to extract from markdown code blocks
        code_match = re.search(
            r"```(?:python)?\s*\n(.*?)```", response_content, re.DOTALL
        )
        if code_match:
            code = code_match.group(1).strip()
            logger.info("Extracted code from markdown code block")
        else:
            # If no code block, assume the entire response is code
            code = response_content.strip()
            logger.info("Using entire response as code (no code block found)")

        if not code:
            logger.warning("No code extracted from response, using empty string")
            code = "# No code generated"

        # CRITICAL: Replace plt.show() with plt.savefig(plot_filename)
        # This ensures plots are saved to files instead of being displayed
        # This handles various formats: plt.show(), plt.show( ), plt.show(block=True), etc.
        if re.search(r"plt\.show\s*\(", code, re.IGNORECASE):
            logger.info("Replacing plt.show() with plt.savefig(plot_filename)")
            code = re.sub(
                r"plt\.show\s*\([^)]*\)",
                "plt.savefig(plot_filename)",
                code,
                flags=re.IGNORECASE,
            )

        # Validate: Check for hardcoded plot filenames (common mistake)
        # Look for plt.savefig with string literals instead of plot_filename variable
        hardcoded_patterns = [
            r'plt\.savefig\s*\(\s*["\'][^"\']+\.(png|jpg|jpeg|pdf)["\']',  # Hardcoded filename
            r'plt\.savefig\s*\(\s*["\'][^"\']+["\']',  # Any hardcoded string
        ]
        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
            if matches:
                logger.warning(
                    f"⚠️  WARNING: CodeAgent generated hardcoded plot filename(s)! "
                    f"Found: {matches}. "
                    f"The code should use 'plot_filename' variable instead. "
                    f"This may cause plots to be saved in the wrong location."
                )
                # Try to fix common cases by replacing with plot_filename
                # But only if plot_filename is not already used
                if "plot_filename" not in code:
                    logger.warning(
                        "Attempting to fix: replacing hardcoded savefig calls with plot_filename"
                    )
                    # Replace plt.savefig("filename.png") with plt.savefig(plot_filename)
                    code = re.sub(
                        r'plt\.savefig\s*\(\s*["\'][^"\']+\.(png|jpg|jpeg|pdf)["\']\s*\)',
                        "plt.savefig(plot_filename)",
                        code,
                        flags=re.IGNORECASE,
                    )

        # Store code in state for ToolAgent to use
        code_msg = AIMessage(content=f"CODE_TO_EXECUTE: {code}")
        logger.info(f"Generated code (length: {len(code)} characters)")

        self.log_node_end()
        return {
            "messages": [code_msg],
            "generated_code": code,
        }
