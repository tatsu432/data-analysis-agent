"""Verifier node for verifying responses."""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ..prompts import VERIFIER_PROMPT
from .base import BaseNode
from .utils import extract_content_text, is_tool_name

logger = logging.getLogger(__name__)

MAX_VERIFICATION_RETRIES = 3


class VerifierNode(BaseNode):
    """Verify if the agent's response is sufficient to answer the user's query."""

    def __init__(self, llm_json: BaseChatModel):
        """Initialize the verifier node.

        Args:
            llm_json: Low-temperature LLM for structured output
        """
        super().__init__("verifier")
        self.llm_json = llm_json

    def __call__(self, state: dict) -> dict:
        """Verify the response."""
        self.log_node_start()
        messages = state["messages"]
        intent = state.get("intent", "ANALYSIS")
        query_classification = state.get("query_classification", "DATA_ANALYSIS")
        verification_retry_count = state.get("verification_retry_count", 0)

        # If router selected KNOWLEDGE, CONFLUENCE, or OTHER, don't apply DATA_ANALYSIS rules
        # Only apply DATA_ANALYSIS verification rules when intent is ANALYSIS
        if intent != "ANALYSIS":
            # Override query_classification for non-ANALYSIS intents
            if intent == "KNOWLEDGE":
                query_classification = "DOCUMENT_QA"
                logger.info(
                    "Router intent is KNOWLEDGE - overriding query_classification to DOCUMENT_QA (run_analysis not required)"
                )
            elif intent == "CONFLUENCE":
                query_classification = "DOCUMENT_QA"
                logger.info(
                    "Router intent is CONFLUENCE - overriding query_classification to DOCUMENT_QA (run_analysis not required)"
                )
            else:  # OTHER
                query_classification = "DOCUMENT_QA"
                logger.info(
                    "Router intent is OTHER - overriding query_classification to DOCUMENT_QA (run_analysis not required)"
                )
        else:
            logger.info(
                f"Router intent is ANALYSIS - using query_classification: {query_classification}"
            )

        if verification_retry_count >= MAX_VERIFICATION_RETRIES:
            logger.warning(
                f"Maximum verification retries ({MAX_VERIFICATION_RETRIES}) reached. Accepting response."
            )
            return {
                "verification_result": {
                    "is_sufficient": True,
                    "reason": "Maximum retries reached",
                    "feedback": "",
                }
            }

        # Get the original user query
        user_query = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content if hasattr(msg, "content") else str(msg)
                break

        if not user_query:
            # No user query found, accept the response
            logger.warning("No user query found, accepting response")
            return {
                "verification_result": {
                    "is_sufficient": True,
                    "reason": "No user query found",
                    "feedback": "",
                }
            }

        # Check if run_analysis was called (for DATA_ANALYSIS queries)
        has_run_analysis = False
        has_tool_calls = False

        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    has_tool_calls = True
                    for tool_call in msg.tool_calls:
                        tool_name = getattr(tool_call, "name", "")
                        if is_tool_name(tool_name, "run_analysis"):
                            has_run_analysis = True
                            break
                break

        # Check for run_analysis in tool messages
        if not has_run_analysis:
            for msg in reversed(messages):
                if hasattr(msg, "name") and is_tool_name(msg.name, "run_analysis"):
                    has_run_analysis = True
                    break

        # For DATA_ANALYSIS queries, check if run_analysis was called
        if query_classification in ["DATA_ANALYSIS", "BOTH"]:
            if not has_run_analysis:
                # Check if we only have list_datasets or get_dataset_schema calls
                only_listing_tools = True
                for msg in reversed(messages):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = getattr(tool_call, "name", "")
                            if not (
                                is_tool_name(tool_name, "list_datasets")
                                or is_tool_name(tool_name, "get_dataset_schema")
                            ):
                                only_listing_tools = False
                                break
                        if not only_listing_tools:
                            break

                if only_listing_tools and has_tool_calls:
                    logger.warning(
                        "Verification failed: Only listing tools called, no run_analysis"
                    )
                    return {
                        "verification_result": {
                            "is_sufficient": False,
                            "reason": "Only dataset listing tools were called, but no actual analysis was performed",
                            "feedback": "You only listed datasets or examined schemas but did not execute the analysis. You MUST call run_analysis to perform the actual data analysis and answer the user's question. The workflow is not complete until you have executed run_analysis and received actual results.",
                        },
                        "verification_retry_count": verification_retry_count + 1,
                    }

        # Use LLM to verify the response
        # Get recent conversation context (last few messages)
        recent_messages_raw = messages[-10:]  # Last 10 messages for context
        recent_messages = []
        tool_results_summary = []

        for msg in recent_messages_raw:
            if isinstance(msg, ToolMessage):
                # Convert tool messages to text summaries to avoid pairing issues
                tool_name = getattr(msg, "name", "tool")
                tool_content = extract_content_text(getattr(msg, "content", ""))
                tool_results_summary.append(f"[{tool_name}]: {tool_content[:300]}")
            elif (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                # AI message with tool_calls - convert to text summary but preserve AI role
                ai_content = extract_content_text(getattr(msg, "content", ""))
                if ai_content:
                    recent_messages.append(
                        AIMessage(content=f"[AI called tools]: {ai_content[:500]}")
                    )
                else:
                    # No content, just tool calls - create a summary
                    tool_names = [getattr(tc, "name", "tool") for tc in msg.tool_calls]
                    recent_messages.append(
                        AIMessage(content=f"[AI called tools: {', '.join(tool_names)}]")
                    )
            else:
                # Human messages and AI messages without tool_calls - keep as-is
                recent_messages.append(msg)

        # Add tool results summary to system message if any were filtered
        system_content = f"User's original query: {user_query}\n\nRouter intent: {intent}\n\nQuery classification: {query_classification}\n\nHas run_analysis been called: {has_run_analysis}\n\nReview the conversation and determine if the response adequately answers the user's question."
        if tool_results_summary:
            system_content += "\n\nTool Results Summary:\n" + "\n".join(
                tool_results_summary
            )

        # Create verification prompt
        verification_messages = [
            SystemMessage(content=system_content)
        ] + recent_messages

        prompt = VERIFIER_PROMPT.invoke({"messages": verification_messages})
        logger.info("Invoking verifier LLM...")

        # Try up to 2 times to get valid JSON
        max_parse_attempts = 2
        verification_data = None
        response_text = None

        try:
            for attempt in range(max_parse_attempts):
                if attempt > 0:
                    logger.warning(
                        f"Retrying verifier JSON parsing (attempt {attempt + 1}/{max_parse_attempts})"
                    )
                    # Add a more explicit instruction for retry
                    retry_system_msg = SystemMessage(
                        content="CRITICAL: You MUST output ONLY a valid JSON object. No markdown, no code blocks, no text before or after. Just the JSON object starting with { and ending with }."
                    )
                    retry_messages = [retry_system_msg] + verification_messages[
                        1:
                    ]  # Keep the original system message context
                    retry_prompt = VERIFIER_PROMPT.invoke({"messages": retry_messages})
                    response = self.llm_json.invoke(retry_prompt.messages)
                else:
                    response = self.llm_json.invoke(prompt.messages)

                response_text = extract_content_text(getattr(response, "content", None))
                if response_text:
                    response_text = response_text.strip()

                # Log the raw response for debugging (only on first attempt to avoid spam)
                if attempt == 0:
                    logger.info(
                        f"Verifier raw response (first 500 chars): {response_text[:500]}"
                    )

                # Try to extract JSON from markdown code blocks if present
                json_match = re.search(
                    r"```(?:json)?\s*\n?({.*?})\n?```", response_text, re.DOTALL
                )
                if json_match:
                    response_text = json_match.group(1).strip()
                    logger.info("Extracted JSON from markdown code block")

                # Also try to extract JSON object directly if wrapped in text
                if not json_match:
                    # More robust pattern to find JSON object
                    json_match = re.search(
                        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\"is_sufficient\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
                        response_text,
                        re.DOTALL,
                    )
                    if json_match:
                        response_text = json_match.group(0).strip()
                        logger.info("Extracted JSON object from text")

                # Try to find JSON object by looking for the structure
                if not json_match:
                    # Look for JSON object that contains is_sufficient
                    brace_start = response_text.find("{")
                    if brace_start >= 0:
                        # Try to find matching closing brace
                        brace_count = 0
                        for i in range(brace_start, len(response_text)):
                            if response_text[i] == "{":
                                brace_count += 1
                            elif response_text[i] == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    potential_json = response_text[brace_start : i + 1]
                                    if '"is_sufficient"' in potential_json:
                                        response_text = potential_json
                                        logger.info(
                                            "Extracted JSON by finding matching braces"
                                        )
                                        break

                # Parse JSON response
                try:
                    verification_data = json.loads(response_text)
                    logger.info("Successfully parsed verification JSON")
                    break  # Success, exit retry loop
                except (ValueError, json.JSONDecodeError, TypeError) as parse_error:
                    if attempt < max_parse_attempts - 1:
                        logger.warning(
                            f"JSON parse failed on attempt {attempt + 1}: {parse_error}. Will retry..."
                        )
                        continue
                    else:
                        # Last attempt failed, will handle in except block below
                        raise

            # If we successfully parsed JSON, process it
            if verification_data is not None:
                is_sufficient = verification_data.get("is_sufficient", True)
                reason = verification_data.get("reason", "")
                feedback = verification_data.get("feedback", "")

                logger.info(f"Verification result: is_sufficient={is_sufficient}")
                logger.info(f"Reason: {reason}")
                if feedback:
                    logger.info(f"Feedback: {feedback}")

                # If verification failed, add feedback message to guide the agent
                result_updates = {
                    "verification_result": {
                        "is_sufficient": is_sufficient,
                        "reason": reason,
                        "feedback": feedback,
                    },
                    "verification_retry_count": (
                        verification_retry_count + 1 if not is_sufficient else 0
                    ),
                }

                if not is_sufficient and feedback:
                    # Check if feedback message already exists to avoid duplicates
                    has_feedback = False
                    for msg in messages:
                        if (
                            isinstance(msg, SystemMessage)
                            and "VERIFICATION FEEDBACK" in msg.content
                        ):
                            has_feedback = True
                            break

                    if not has_feedback:
                        # Add feedback message to guide the agent
                        feedback_msg = SystemMessage(
                            content=f"VERIFICATION FEEDBACK: {feedback}\n\nYou must replan and continue the analysis. Do not stop until you have completed the full workflow."
                        )
                        # Add feedback to messages
                        current_messages = list(messages)
                        current_messages.insert(0, feedback_msg)
                        result_updates["messages"] = current_messages

                self.log_node_end()
                return result_updates

            # If we get here, JSON parsing failed after all retries
            raise ValueError("Failed to parse JSON after all retry attempts")

        except (ValueError, json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"Failed to parse verification JSON response after all attempts: {e}"
            )
            if response_text:
                logger.error(f"Full response text: {response_text}")

            # Try to extract meaningful information from the response even if JSON parsing fails
            is_sufficient_heuristic = True
            reason_heuristic = "Could not parse verification response"

            # Try to infer from response text if available
            if response_text:
                response_lower = response_text.lower()
                if any(
                    word in response_lower
                    for word in [
                        "insufficient",
                        "not sufficient",
                        "missing",
                        "incomplete",
                        "failed",
                    ]
                ):
                    is_sufficient_heuristic = False
                    reason_heuristic = (
                        "Verifier indicated response is insufficient (parsed from text)"
                    )
                elif any(
                    word in response_lower
                    for word in ["sufficient", "complete", "adequate", "answered"]
                ):
                    is_sufficient_heuristic = True
                    reason_heuristic = (
                        "Verifier indicated response is sufficient (parsed from text)"
                    )

            # Fallback: check if run_analysis was called (only for ANALYSIS intent)
            if (
                intent == "ANALYSIS"
                and query_classification in ["DATA_ANALYSIS", "BOTH"]
                and not has_run_analysis
            ):
                return {
                    "verification_result": {
                        "is_sufficient": False,
                        "reason": "run_analysis was not called - analysis incomplete",
                        "feedback": "You must call run_analysis to perform the actual data analysis. The workflow is not complete until you have executed run_analysis and received actual results.",
                    },
                    "verification_retry_count": verification_retry_count + 1,
                }
            else:
                # Use heuristic if available, otherwise default based on run_analysis status
                if has_run_analysis:
                    return {
                        "verification_result": {
                            "is_sufficient": is_sufficient_heuristic,
                            "reason": f"{reason_heuristic}. run_analysis was called, so assuming response may be sufficient.",
                            "feedback": ""
                            if is_sufficient_heuristic
                            else "Please verify the response addresses the user's question completely.",
                        }
                    }
                else:
                    return {
                        "verification_result": {
                            "is_sufficient": is_sufficient_heuristic,
                            "reason": f"{reason_heuristic}. Unable to verify due to parsing error.",
                            "feedback": ""
                            if is_sufficient_heuristic
                            else "Please verify the response addresses the user's question completely.",
                        }
                    }
