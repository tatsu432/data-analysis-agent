"""Code generation node."""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from ..prompts import CODE_GENERATION_PROMPT
from .base import BaseNode
from .utils import extract_content_text

logger = logging.getLogger(__name__)


class CodeGenerationNode(BaseNode):
    """Generate Python code for data analysis using the coding LLM."""

    def __init__(
        self,
        coding_llm_with_tools: BaseChatModel,
        coding_tools: list,
        coding_tool_choice: str | None = None,
    ):
        """Initialize the code generation node.

        Args:
            coding_llm_with_tools: Coding LLM with tools bound
            coding_tools: List of coding tools
            coding_tool_choice: Tool choice setting
        """
        super().__init__("code_generation")
        self.coding_llm_with_tools = coding_llm_with_tools
        self.coding_tools = coding_tools
        self.coding_tool_choice = coding_tool_choice

    def __call__(self, state: dict) -> dict:
        """Generate code for data analysis."""
        self.log_node_start()
        messages = state["messages"]

        # Extract dataset IDs from tool responses
        dataset_ids_found = set()
        schemas_found = {}  # Track which datasets have schema information

        # Look through messages for tool responses that contain dataset information
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "")

                # Extract dataset_id from get_dataset_schema responses
                if tool_name == "get_dataset_schema":
                    # Try to extract dataset_id from the tool call that preceded this response
                    # Look for the tool call in previous messages
                    for prev_msg in reversed(messages[: messages.index(msg)]):
                        if hasattr(prev_msg, "tool_calls") and prev_msg.tool_calls:
                            for tool_call in prev_msg.tool_calls:
                                if (
                                    getattr(tool_call, "name", "")
                                    == "get_dataset_schema"
                                ):
                                    args = getattr(tool_call, "args", {})
                                    if isinstance(args, dict):
                                        # Handle both direct args and nested payload
                                        dataset_id = args.get("dataset_id") or args.get(
                                            "payload", {}
                                        ).get("dataset_id")
                                        if dataset_id:
                                            dataset_ids_found.add(dataset_id)
                                            # Mark that we have schema for this dataset
                                            schemas_found[dataset_id] = True
                                            break

                # Extract dataset IDs from list_datasets responses
                elif tool_name == "list_datasets":
                    try:
                        # Try to parse the tool response content
                        content = getattr(msg, "content", "")
                        data = None

                        if isinstance(content, dict):
                            # Content is already a dict
                            data = content
                        elif isinstance(content, str):
                            # Try to parse as JSON
                            try:
                                data = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                pass

                        if isinstance(data, dict) and "datasets" in data:
                            for dataset in data["datasets"]:
                                if isinstance(dataset, dict) and "id" in dataset:
                                    dataset_ids_found.add(dataset["id"])
                    except Exception as e:
                        logger.debug(
                            f"Error extracting dataset IDs from list_datasets response: {e}"
                        )

        # Extract the user's original question for clarity
        user_question = None
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content if hasattr(msg, "content") else str(msg)
                break

        # Extract error context from CODE_GENERATION_NEEDED messages
        error_context = None
        previous_code = None
        filtered_messages = []
        for msg in messages:
            # Extract error context from CODE_GENERATION_NEEDED routing messages
            if (
                hasattr(msg, "content")
                and isinstance(msg.content, str)
                and "CODE_GENERATION_NEEDED" in msg.content
            ):
                # Extract error information if present
                content = msg.content
                if "ERROR CONTEXT:" in content:
                    error_context = content.split("ERROR CONTEXT:")[-1].strip()
                if "Error:" in content:
                    error_part = content.split("Error:")[-1].split("\n")[0].strip()
                    if error_context is None:
                        error_context = error_part
                # Don't include the routing message itself in filtered messages
                continue
            filtered_messages.append(msg)

        # Also extract previous code from tool calls
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if getattr(tool_call, "name", "") == "run_analysis":
                        args = getattr(tool_call, "args", {})
                        if isinstance(args, dict):
                            previous_code = args.get("code", "")
                            break
                if previous_code:
                    break

        # Add knowledge context if available
        prompt_messages = list(filtered_messages)
        knowledge_context = state.get("knowledge_context", "")

        # Build system message with context
        system_parts = []

        # Add error context if this is a retry after an error
        if error_context:
            logger.info(
                f"Code generation retry with error context: {error_context[:200]}..."
            )
            error_guidance = f"""🚨 PREVIOUS CODE EXECUTION FAILED - FIX REQUIRED

The previous code execution failed with an error. You MUST fix the code before calling run_analysis again.

ERROR MESSAGE:
{error_context[:800]}

PREVIOUS CODE (that failed):
{previous_code[:1000] if previous_code else "No previous code found"}

YOUR TASK:
1. Carefully analyze the error message to understand what went wrong
2. Identify the specific issue (wrong column name, syntax error, logic error, etc.)
3. Fix the code based on the error
4. Call run_analysis with the corrected code

COMMON FIXES:
- KeyError/NameError: Check column names against schema - use exact names from get_dataset_schema responses
- SyntaxError: Fix Python syntax errors
- TypeError: Check data types and conversions
- ValueError: Check filter conditions and data ranges
- Empty result: Check if filters are too restrictive

CRITICAL: You MUST fix the error before calling run_analysis. Do NOT repeat the same mistake."""
            system_parts.append(error_guidance)

        if (
            knowledge_context
            and knowledge_context.strip()
            and len(knowledge_context.strip()) > 10
        ):
            system_parts.append(
                f"Document knowledge context:\n{knowledge_context}\n\nUse this context to understand domain terms and map them to dataset columns when writing code."
            )

        # Add the user's question prominently if available
        if user_question:
            system_parts.append(
                f"USER'S QUESTION (THIS IS WHAT YOU NEED TO ANSWER):\n{user_question}\n"
            )

        # Add dataset IDs hint if found
        if dataset_ids_found:
            dataset_ids_list = sorted(list(dataset_ids_found))
            datasets_with_schema = [
                ds_id for ds_id in dataset_ids_list if schemas_found.get(ds_id, False)
            ]
            datasets_without_schema = [
                ds_id
                for ds_id in dataset_ids_list
                if not schemas_found.get(ds_id, False)
            ]

            schema_info = []
            if datasets_with_schema:
                schema_info.append(
                    f"Datasets with schema information (you can use these directly): {', '.join(datasets_with_schema)}"
                )
            if datasets_without_schema:
                schema_info.append(
                    f"CRITICAL - Datasets WITHOUT schema information: {', '.join(datasets_without_schema)}\n"
                    f"  - You MUST call get_dataset_schema() for these datasets BEFORE generating code\n"
                    f"  - Do NOT guess column names - this will cause KeyError\n"
                    f"  - Example: Call get_dataset_schema('{datasets_without_schema[0]}') first, then use the schema to write correct code"
                )

            schema_info_text = "\n".join(schema_info) if schema_info else ""
            system_parts.append(
                f"CRITICAL - Datasets identified in conversation:\n"
                f"The following dataset IDs have been mentioned or examined: {', '.join(dataset_ids_list)}\n"
                f"{schema_info_text}\n"
                f"When you generate code that uses these datasets (e.g., dfs['{dataset_ids_list[0]}']), "
                f"you MUST include ALL referenced dataset IDs in the dataset_ids parameter of run_analysis.\n"
                f"Example: If your code uses dfs['{dataset_ids_list[0]}'], then dataset_ids=['{dataset_ids_list[0]}']\n"
            )

            if datasets_without_schema:
                system_parts.append(
                    f"\n⚠️ WARNING: The main agent has not yet retrieved schema for these datasets: {', '.join(datasets_without_schema)}\n"
                    f"However, you MUST still generate code and call run_analysis. Use common column name patterns or check the conversation history for any schema hints.\n"
                    f"The main agent should have gathered this information, but proceed with code generation anyway."
                )
            else:
                system_parts.append(
                    "\n✅ YOU HAVE ALL SCHEMA INFORMATION. Generate code using the schema information from the conversation history and call run_analysis immediately."
                )

            # Add explicit instruction about using schema from conversation
            if datasets_with_schema:
                system_parts.append(
                    f"\n✅ Schema information is available in the conversation history for: {', '.join(datasets_with_schema)}\n"
                    f"Use the schema information from get_dataset_schema() responses in the conversation to generate correct code.\n"
                    f"Then call run_analysis with your generated code."
                )
        else:
            # Even if no dataset IDs found, we should still try to generate code
            system_parts.append(
                "NOTE: Generate code based on the user's question and available context from the conversation history.\n"
                "The main agent has gathered all necessary information. You MUST call run_analysis with your generated code.\n"
                "DO NOT refuse - generate code and execute it."
            )

        if system_parts:
            system_content = "\n\n".join(system_parts)
            prompt_messages.insert(0, SystemMessage(content=system_content))
            logger.info(
                f"Injected system context with {len(dataset_ids_found)} dataset ID(s): {sorted(dataset_ids_found)}"
            )

        prompt = CODE_GENERATION_PROMPT.invoke({"messages": prompt_messages})
        logger.info("Invoking coding LLM for code generation...")
        logger.info(
            f"Available coding tools: {[tool.name for tool in self.coding_tools]}"
        )
        logger.info(f"Tool choice setting: {self.coding_tool_choice}")
        logger.info(f"Number of messages in prompt: {len(prompt.messages)}")
        response = self.coding_llm_with_tools.invoke(prompt.messages)

        response_content_text = extract_content_text(getattr(response, "content", None))

        # Validate response - check if model output natural language instead of tool calls
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
        has_content = response_content_text and response_content_text.strip()

        # Detect refusal patterns in the response
        refusal_keywords = [
            "cannot",
            "can't",
            "unable",
            "refuse",
            "not able",
            "don't have",
            "need more",
            "insufficient",
            "lack of",
            "missing information",
            "not enough",
            "requires more",
            "cannot proceed",
            "unable to",
        ]
        is_refusal = False
        if has_content:
            content_lower = response_content_text.lower()
            is_refusal = any(keyword in content_lower for keyword in refusal_keywords)

        if not has_tool_calls and has_content:
            content_preview = (
                response_content_text[:200]
                if len(response_content_text) > 200
                else response_content_text
            )

            if is_refusal:
                logger.error(
                    f"Code generation LLM REFUSED to make tool calls. "
                    f"Refusal message: {content_preview}..."
                )
                logger.error(
                    "This is likely because the model doesn't have enough context or information. "
                    "Attempting to provide additional context and retry..."
                )

                # Try to provide more context and retry once
                retry_system_msg = SystemMessage(
                    content=f"""CRITICAL: You MUST make a tool call. Even if you're uncertain, you MUST still call run_analysis with the best code you can generate.

The previous response was a refusal, which is not allowed. You MUST make a tool call.

If you're missing information:
1. Use the dataset IDs that were mentioned in the conversation: {sorted(list(dataset_ids_found)) if dataset_ids_found else "none found"}
2. Generate code based on the user's question and available context
3. Even if the code might not be perfect, you MUST call run_analysis

EXAMPLE - You MUST do this:
Call run_analysis with:
- code: "result_df = dfs['dataset_id'].head(10)"  # Replace with actual analysis code
- dataset_ids: ['dataset_id']  # Use actual dataset IDs from conversation

DO NOT refuse. DO NOT explain why you can't. Just make the tool call."""
                )

                # Insert the retry message before the last system message (if any) or at the beginning
                retry_prompt_messages = list(prompt_messages)
                # Find the last SystemMessage and insert after it, or insert at the beginning
                insert_idx = 0
                for i, msg in enumerate(retry_prompt_messages):
                    if isinstance(msg, SystemMessage):
                        insert_idx = i + 1
                retry_prompt_messages.insert(insert_idx, retry_system_msg)

                retry_prompt = CODE_GENERATION_PROMPT.invoke(
                    {"messages": retry_prompt_messages}
                )
                logger.info("Retrying code generation with additional context...")
                response = self.coding_llm_with_tools.invoke(retry_prompt.messages)

                # Re-check after retry
                response_content_text = extract_content_text(
                    getattr(response, "content", None)
                )
                has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
                has_content = response_content_text and response_content_text.strip()
                if has_content:
                    content_lower = response_content_text.lower()
                    is_refusal = any(
                        keyword in content_lower for keyword in refusal_keywords
                    )

                if not has_tool_calls and is_refusal:
                    logger.error(
                        "Code generation LLM still refused after retry with additional context. "
                        "This indicates a deeper issue - the model may not have sufficient information "
                        "or there may be a configuration problem."
                    )
                    logger.error(f"Full refusal message: {response_content_text}")
                    logger.error(
                        f"Available dataset IDs from conversation: {sorted(list(dataset_ids_found)) if dataset_ids_found else 'none'}"
                    )
                    logger.error(
                        f"Number of messages in context: {len(prompt_messages)}"
                    )
            else:
                logger.error(
                    f"Code generation LLM output natural language instead of tool calls. "
                    f"Content preview: {content_preview}..."
                )
                logger.error(
                    "This indicates the model did not follow instructions to make tool calls. "
                    "Check tool_choice setting and prompt."
                )

            # Try to extract code from the natural language response and create a tool call
            code_match = re.search(
                r"```(?:python)?\s*\n(.*?)```", response_content_text, re.DOTALL
            )
            if code_match:
                extracted_code = code_match.group(1).strip()
                logger.warning(
                    "Attempting to extract code from natural language response"
                )
                # Try to extract dataset IDs from the extracted code
                pattern = r"dfs\[['\"]([^'\"]+)['\"]\]"
                code_dataset_ids = set(re.findall(pattern, extracted_code))

                if code_dataset_ids:
                    # Create a synthetic tool call
                    tool_call = ToolCall(
                        name="run_analysis",
                        args={
                            "code": extracted_code,
                            "dataset_ids": sorted(list(code_dataset_ids)),
                        },
                        id="extracted_from_natural_language",
                    )
                    response.tool_calls = [tool_call]
                    response.content = ""  # Clear the natural language content
                    logger.warning(
                        f"Extracted code and created synthetic tool call with dataset_ids: {sorted(code_dataset_ids)}"
                    )
                else:
                    # Fallback to dataset_ids from conversation
                    if dataset_ids_found:
                        tool_call = ToolCall(
                            name="run_analysis",
                            args={
                                "code": extracted_code,
                                "dataset_ids": sorted(list(dataset_ids_found)),
                            },
                            id="extracted_from_natural_language",
                        )
                        response.tool_calls = [tool_call]
                        response.content = ""
                        logger.warning(
                            f"Extracted code and used dataset_ids from conversation: {sorted(dataset_ids_found)}"
                        )
                    else:
                        logger.error(
                            "Could not extract code or determine dataset_ids. "
                            "Response will be passed through but may cause errors."
                        )
                        # If this was a refusal and we couldn't extract code, create a minimal tool call
                        if is_refusal and dataset_ids_found:
                            logger.warning(
                                "Creating minimal tool call with placeholder code to prevent workflow failure. "
                                "This is a fallback - the code may need to be fixed by the main agent."
                            )
                            # Create a minimal code that at least loads the dataset
                            placeholder_code = f"# Placeholder code - model refused to generate proper code\nresult_df = dfs['{list(dataset_ids_found)[0]}'].head(1)"
                            tool_call = ToolCall(
                                name="run_analysis",
                                args={
                                    "code": placeholder_code,
                                    "dataset_ids": sorted(list(dataset_ids_found)),
                                },
                                id="fallback_after_refusal",
                            )
                            response.tool_calls = [tool_call]
                            response.content = ""  # Clear the refusal message
                            logger.warning(
                                f"Created fallback tool call with dataset_ids: {sorted(list(dataset_ids_found))}"
                            )

        # Log tool calls if any and validate/fix them
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(
                f"Coding LLM requested {len(response.tool_calls)} tool call(s):"
            )

            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, "name", "unknown")
                logger.info(f"  - {tool_name}")

                # Validate and fix run_analysis tool calls
                if tool_name == "run_analysis":
                    args = getattr(tool_call, "args", {})
                    if not isinstance(args, dict):
                        args = {}

                    # Remove None values from args (Pydantic rejects explicit None for optional fields)
                    # This fixes the issue where primary_dataset_id=None causes validation errors
                    args = {k: v for k, v in args.items() if v is not None}
                    tool_call.args = args  # Update the tool call with filtered args

                    # Check if dataset_ids is missing
                    if "dataset_ids" not in args or not args.get("dataset_ids"):
                        code = args.get("code", "")
                        if code:
                            # Extract dataset IDs from code using regex
                            pattern = r"dfs\[['\"]([^'\"]+)['\"]\]"
                            code_dataset_ids = set(re.findall(pattern, code))

                            if code_dataset_ids:
                                args["dataset_ids"] = sorted(list(code_dataset_ids))
                                tool_call.args = args
                                logger.warning(
                                    f"Fixed missing dataset_ids in run_analysis call. "
                                    f"Extracted from code: {args['dataset_ids']}"
                                )
                            elif dataset_ids_found:
                                # Fallback to dataset IDs found in conversation
                                args["dataset_ids"] = sorted(list(dataset_ids_found))
                                tool_call.args = args
                                logger.warning(
                                    f"Fixed missing dataset_ids in run_analysis call. "
                                    f"Using dataset IDs from conversation: {args['dataset_ids']}"
                                )
                            else:
                                logger.error(
                                    "run_analysis call is missing dataset_ids and could not be extracted from code or conversation."
                                )
        else:
            logger.warning(
                "Coding LLM response has no tool calls - code generation may have failed"
            )

        logger.info("Code generation response generated")
        self.log_node_end()
        return {"messages": [response]}
