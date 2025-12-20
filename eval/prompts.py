"""
Prompt templates for evaluation metrics.

This module contains prompt templates used by various evaluation metrics,
particularly for LLM-as-judge evaluations with checklist-based criteria.
"""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

from .schema import Criterion

# LLM Judge Prompt Template for Checklist-based Evaluation
LLM_JUDGE_CHECKLIST_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert evaluator assessing the quality of an AI agent's response using a checklist of criteria.",
        ),
        (
            "human",
            """**User Query:**
{query}

**Agent Response:**
{agent_response}

{reference_section}

**Evaluation Checklist:**
{criteria_list}

For each criterion, evaluate the response and provide:
- A satisfaction score from 0 to 100 (where 100 means fully satisfied, 0 means not satisfied at all)
- A brief reasoning explaining your score

Respond with ONLY a valid JSON object in this exact format:
{{
  "criterion_results": [
    {{
      "criterion_name": "name_of_criterion",
      "satisfaction_score": 85.0,
      "reasoning": "Brief explanation of the score"
    }},
    ...
  ]
}}

Do not include any additional text, explanations, or markdown formatting outside the JSON object.""",
        ),
    ]
)


def format_criteria_list(criteria: List[Criterion]) -> str:
    """
    Format a list of criteria into a readable checklist string.

    Args:
        criteria: List of Criterion objects

    Returns:
        Formatted string for the prompt
    """
    lines = []
    for idx, criterion in enumerate(criteria, start=1):
        weight_note = (
            f" (weight: {criterion.weight})" if criterion.weight != 1.0 else ""
        )
        lines.append(
            f"{idx}. **{criterion.name}**{weight_note}\n   {criterion.description}"
        )
    return "\n\n".join(lines)


def build_llm_judge_messages(
    query: str,
    agent_response: str,
    criteria: List[Criterion],
    reference: Optional[str] = None,
):
    """
    Build messages for LLM-as-judge evaluation using ChatPromptTemplate.

    Args:
        query: The original user query
        agent_response: The agent's response to evaluate
        criteria: List of criteria to evaluate (checklist)
        reference: Optional reference/expected answer for comparison

    Returns:
        List of formatted messages ready for LLM invocation
    """
    reference_section = ""
    if reference:
        reference_section = f"\n**Reference/Expected Answer:**\n{reference}\n"

    criteria_list = format_criteria_list(criteria)

    # Format the template with the provided values
    messages = LLM_JUDGE_CHECKLIST_PROMPT_TEMPLATE.format_messages(
        query=query,
        agent_response=agent_response,
        reference_section=reference_section,
        criteria_list=criteria_list,
    )

    return messages
