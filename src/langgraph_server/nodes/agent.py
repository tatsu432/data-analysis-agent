"""Agent node for main reasoning."""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ..prompts import ANALYSIS_PROMPT
from .base import BaseNode

logger = logging.getLogger(__name__)

MAX_CODE_GENERATION_RETRIES = 3


class AgentNode(BaseNode):
    """Call the LLM with the current state."""

    def __init__(self, llm_with_tools: BaseChatModel):
        """Initialize the agent node.

        Args:
            llm_with_tools: LLM with tools bound
        """
        super().__init__("agent")
        self.llm_with_tools = llm_with_tools

    def __call__(self, state: dict) -> dict:
        """Call the LLM with the current state."""
        self.log_node_start()
        messages = state["messages"]
        logger.info(f"Number of messages in state: {len(messages)}")
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content") and last_message.content:
                logger.debug(
                    f"Last message content: {str(last_message.content)[:200]}..."
                )
        # Add knowledge context if available and non-empty
        prompt_messages = list(messages)
        knowledge_context = state.get("knowledge_context", "")
        if (
            knowledge_context
            and knowledge_context.strip()
            and len(knowledge_context.strip()) > 10
        ):
            # Insert knowledge context at the beginning
            knowledge_msg = SystemMessage(
                content=f"Document knowledge context:\n{knowledge_context}\n\nUse this context ONLY to understand domain terms and map them to dataset columns. Do NOT let this distract from your primary task of data analysis."
            )
            prompt_messages.insert(0, knowledge_msg)

        # Check if we have tool responses but haven't called run_analysis yet
        # This helps prevent Qwen (and other models) from stopping after list_datasets
        has_tool_responses = False
        has_run_analysis = False
        run_analysis_error = None
        code_generation_retry_count = state.get("code_generation_retry_count", 0)

        # Check for run_analysis errors and extract error message
        for msg in reversed(messages):
            # Check for tool responses (ToolMessage)
            if isinstance(msg, ToolMessage):
                has_tool_responses = True
                tool_name = getattr(msg, "name", "")
                if tool_name == "run_analysis":
                    has_run_analysis = True
                    # Check if there's an error in the result
                    content = getattr(msg, "content", "")
                    if content:
                        try:
                            if isinstance(content, str):
                                result_data = json.loads(content)
                                if isinstance(result_data, dict):
                                    if result_data.get("error") or not result_data.get(
                                        "success", True
                                    ):
                                        error_msg = result_data.get(
                                            "error", "Unknown error"
                                        )
                                        run_analysis_error = error_msg
                                        logger.info(
                                            f"Detected run_analysis error: {error_msg[:200]}..."
                                        )
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, check for error indicators
                            if (
                                '"error"' in str(content).lower()
                                or '"success": false' in str(content).lower()
                            ):
                                run_analysis_error = str(content)
            # Also check for tool calls in AIMessage
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = getattr(tool_call, "name", "")
                    if tool_name == "run_analysis":
                        has_run_analysis = True

        # If we have tool responses but no run_analysis yet, check if we should route to code generation
        # CRITICAL: Only route to code generation if get_dataset_schema has been called
        # The code generation node does NOT have access to get_dataset_schema, so schema must be gathered first
        if has_tool_responses and not has_run_analysis:
            # Check if we have schema information (get_dataset_schema must be called)
            has_schema_info = False
            has_list_datasets = False
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", "")
                    if tool_name == "get_dataset_schema":
                        has_schema_info = True
                        break
                    elif tool_name == "list_datasets":
                        has_list_datasets = True

            if has_schema_info:
                # We have schema info - safe to route to code generation
                logger.info(
                    "Schema information available - routing to code generation node"
                )
                # Return a special marker to indicate code generation is needed
                routing_msg = AIMessage(
                    content="CODE_GENERATION_NEEDED: Dataset schema information has been gathered. Code generation is required to proceed with the analysis."
                )
                return {"messages": [routing_msg]}
            elif has_list_datasets and not has_schema_info:
                # Only list_datasets was called, but no schema yet - remind agent to get schema first
                logger.info(
                    "list_datasets called but get_dataset_schema not yet called - reminding agent to get schema first"
                )

                schema_reminder_msg = SystemMessage(
                    content="IMPORTANT: You have called list_datasets, but you have NOT yet called get_dataset_schema for the datasets you need to analyze. "
                    "CRITICAL: The code generation node does NOT have access to get_dataset_schema. "
                    "You MUST call get_dataset_schema(dataset_id) for each dataset you plan to use BEFORE routing to code generation. "
                    "Without schema information, the code generation will fail because it won't know the column names. "
                    "Please call get_dataset_schema for the relevant dataset(s) now."
                )
                prompt_messages.insert(0, schema_reminder_msg)
                logger.info(
                    "Injected reminder: get_dataset_schema must be called before code generation"
                )

            # Otherwise, inject a reminder (for other tool responses)
            reminder_msg = SystemMessage(
                content="REMINDER: You have received tool responses. "
                "These are INTERMEDIATE INFORMATION, not final answers. "
                "If you need to analyze data, you MUST: "
                "1. Call get_dataset_schema for each dataset you need (if not already done)"
                "2. Route to code generation to execute the actual data analysis"
                "DO NOT stop and present tool responses as your final answer. "
                "The workflow is only complete after code has been generated and run_analysis has been executed."
            )
            prompt_messages.insert(0, reminder_msg)
            logger.info(
                "Injected reminder: tool responses detected but run_analysis not called yet"
            )

        # Check if run_analysis failed and we need to retry code generation
        if (
            run_analysis_error
            and code_generation_retry_count < MAX_CODE_GENERATION_RETRIES
        ):
            logger.info(
                f"run_analysis failed with error (retry {code_generation_retry_count + 1}/{MAX_CODE_GENERATION_RETRIES}). "
                "Analyzing error and routing to code generation for retry."
            )

            # Extract the last code that was executed to help with error analysis
            last_code = None
            for msg in reversed(messages):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if getattr(tool_call, "name", "") == "run_analysis":
                            args = getattr(tool_call, "args", {})
                            if isinstance(args, dict):
                                last_code = args.get("code", "")
                                break
                    if last_code:
                        break

            # Create error analysis message
            error_analysis_msg = SystemMessage(
                content=f"""🚨 CODE EXECUTION ERROR DETECTED - RETRY REQUIRED

The previous code execution failed with an error. You MUST analyze the error and route to code generation with a fix.

ERROR MESSAGE:
{run_analysis_error[:1000]}

PREVIOUS CODE (for reference):
{last_code[:500] if last_code else "No code found"}

YOUR TASK:
1. Analyze the error message to understand what went wrong
2. Identify the root cause (e.g., wrong column name, syntax error, missing import, logic error)
3. Provide clear feedback about what needs to be fixed
4. Route to code generation by indicating CODE_GENERATION_NEEDED with error context

COMMON ERROR TYPES AND FIXES:
- KeyError: Column name doesn't exist - check schema and use correct column names
- SyntaxError: Python syntax error - fix the syntax
- NameError: Undefined variable - check variable names and imports
- TypeError: Wrong data type - check data types and conversions
- ValueError: Invalid value - check filter conditions and data ranges
- Empty result: Filter too restrictive - check date ranges and filter conditions

After analyzing the error, you MUST route to code generation with CODE_GENERATION_NEEDED message that includes:
- What went wrong
- What needs to be fixed
- Any specific guidance for the code generation node

RETRY COUNT: {code_generation_retry_count + 1}/{MAX_CODE_GENERATION_RETRIES}"""
            )
            prompt_messages.insert(0, error_analysis_msg)

            # Update retry count in state (will be returned)
            state_updates = {
                "code_generation_retry_count": code_generation_retry_count + 1
            }
        elif (
            run_analysis_error
            and code_generation_retry_count >= MAX_CODE_GENERATION_RETRIES
        ):
            logger.warning(
                f"Maximum code generation retries ({MAX_CODE_GENERATION_RETRIES}) reached. "
                "Providing error summary to user."
            )
            error_summary_msg = SystemMessage(
                content=f"""⚠️ MAXIMUM RETRIES REACHED

The code generation has been retried {MAX_CODE_GENERATION_RETRIES} times but still fails.

LAST ERROR:
{run_analysis_error[:500]}

You should provide a helpful error message to the user explaining what went wrong and why the analysis could not be completed."""
            )
            prompt_messages.insert(0, error_summary_msg)
            state_updates = {}
        else:
            state_updates = {}

        prompt = ANALYSIS_PROMPT.invoke({"messages": prompt_messages})
        logger.info("Invoking LLM...")
        response = self.llm_with_tools.invoke(prompt.messages)

        # Log tool calls if any
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"LLM requested {len(response.tool_calls)} tool call(s):")
            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                logger.info(f"  - {tool_name}")
        else:
            logger.info("LLM response (no tool calls)")

        # If we're retrying after an error, ensure we route to code generation
        if (
            run_analysis_error
            and code_generation_retry_count < MAX_CODE_GENERATION_RETRIES
        ):
            # Check if the response already indicates code generation is needed
            response_content = getattr(response, "content", "")
            if (
                isinstance(response_content, str)
                and "CODE_GENERATION_NEEDED" not in response_content
            ):
                # Force routing to code generation
                error_context = f"\n\nERROR CONTEXT: {run_analysis_error[:300]}"
                routing_msg = AIMessage(
                    content=f"CODE_GENERATION_NEEDED: Previous code execution failed. Error: {run_analysis_error[:200]}. Please fix the code and retry.{error_context}"
                )
                return {**state_updates, "messages": [routing_msg]}

        self.log_node_end()
        return {**state_updates, "messages": [response]}
