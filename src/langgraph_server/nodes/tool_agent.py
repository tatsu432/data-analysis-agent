"""Tool agent node - only calls run_analysis tool."""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .base import BaseNode
from .utils import extract_content_text, is_tool_name

logger = logging.getLogger(__name__)


class ToolAgentNode(BaseNode):
    """Tool agent that only calls run_analysis."""

    def __init__(self, llm_with_tool: BaseChatModel):
        """Initialize the tool agent node.

        Args:
            llm_with_tool: LLM with run_analysis tool bound (only this tool)
        """
        super().__init__("tool_agent")
        self.llm_with_tool = llm_with_tool

    def __call__(self, state: dict) -> dict:
        """Call run_analysis tool with generated code."""
        self.log_node_start()
        messages = state["messages"]
        generated_code = state.get("generated_code", "")

        # Extract code from messages if not in state
        if not generated_code:
            for msg in reversed(messages):
                content = extract_content_text(getattr(msg, "content", None))
                if "CODE_TO_EXECUTE:" in content:
                    generated_code = content.split("CODE_TO_EXECUTE:")[-1].strip()
                    break

        if not generated_code:
            logger.error("No code found to execute")
            error_msg = AIMessage(
                content="ERROR: No code was generated. Cannot execute analysis."
            )
            return {"messages": [error_msg]}

        # Extract dataset IDs from code
        dataset_ids = set()
        pattern = r'dfs\[["\']([^"\']+)["\']\]'
        code_dataset_ids = set(re.findall(pattern, generated_code))

        # Also check for code aliases
        alias_patterns = [
            (r'df_covid_daily', 'covid_new_cases_daily'),
            (r'df_jpm_patients', 'jpm_patient_data'),
            (r'df_mr_activity', 'mr_activity_data'),
            (r'df_jamdas_patients', 'jamdas_patient_data'),
        ]
        for pattern, dataset_id in alias_patterns:
            if re.search(pattern, generated_code):
                dataset_ids.add(dataset_id)

        dataset_ids.update(code_dataset_ids)

        # Fallback: try to extract from conversation history
        if not dataset_ids:
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if is_tool_name(getattr(tool_call, "name", ""), "get_dataset_schema"):
                            args = getattr(tool_call, "args", {})
                            if isinstance(args, dict):
                                dataset_id = args.get("dataset_id") or args.get("payload", {}).get("dataset_id")
                                if dataset_id:
                                    dataset_ids.add(dataset_id)

        if not dataset_ids:
            logger.warning("No dataset IDs found, using default")
            dataset_ids = {"covid_new_cases_daily"}  # Default fallback

        dataset_ids_list = sorted(list(dataset_ids))

        # Create a message that will trigger tool call
        from langchain_core.messages import SystemMessage, HumanMessage
        system_prompt = f"""You MUST call the run_analysis tool with the following code and dataset IDs.

CODE TO EXECUTE:
{generated_code}

DATASET IDs: {dataset_ids_list}

Call run_analysis with:
- code: The code above
- dataset_ids: {dataset_ids_list}

You MUST make this tool call. Do not provide explanations."""

        tool_messages = [SystemMessage(content=system_prompt), HumanMessage(content="Execute the analysis code now.")]

        logger.info(f"Calling run_analysis with {len(dataset_ids_list)} dataset(s): {dataset_ids_list}")
        logger.info(f"Code length: {len(generated_code)} characters")

        # Invoke LLM to make tool call
        response = self.llm_with_tool.invoke(tool_messages)

        # Validate that tool call was made
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            logger.error("Tool agent did not make tool call, creating synthetic call")
            # Create synthetic tool call
            from langchain_core.messages.tool import ToolCall
            tool_call = ToolCall(
                name="run_analysis",
                args={
                    "code": generated_code,
                    "dataset_ids": dataset_ids_list,
                },
                id="synthetic_tool_call",
            )
            response.tool_calls = [tool_call]
            response.content = ""
        else:
            # Validate and fix tool call args
            for tool_call in response.tool_calls:
                if is_tool_name(getattr(tool_call, "name", ""), "run_analysis"):
                    args = getattr(tool_call, "args", {})
                    if not isinstance(args, dict):
                        args = {}
                    
                    # Ensure code and dataset_ids are set
                    if "code" not in args or not args.get("code"):
                        args["code"] = generated_code
                    if "dataset_ids" not in args or not args.get("dataset_ids"):
                        args["dataset_ids"] = dataset_ids_list
                    
                    tool_call.args = args

        self.log_node_end()
        return {"messages": [response]}

