"""
LLM-as-judge metric system for evaluating agent outputs.

This module provides an LLM-based evaluation metric that uses an LLM to judge
the quality of agent responses based on a checklist of criteria, with each
criterion evaluated separately and scored from 0 to 100.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .langgraph_client import LangGraphResult
from .prompts import build_llm_judge_messages
from .schema import (
    Criterion,
    CriterionResult,
    LLMJudgeConfig,
    MetricResult,
)

# Load .env file from project root (same as main application)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback: try current directory
    load_dotenv()

logger = logging.getLogger("eval.metrics")

# Default LLM for judge metrics (can be overridden via env var)
DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_JUDGE_PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER", "openai")


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        query: str,
        result: LangGraphResult,
        reference: Optional[str] = None,
    ) -> MetricResult:
        """
        Evaluate the agent result against this metric.

        Args:
            query: The original user query
            result: The agent's response result
            reference: Optional reference answer or expected output

        Returns:
            MetricResult with score and details
        """
        pass


class LLMJudgeMetric(BaseMetric):
    """
    LLM-as-judge metric that evaluates output quality using an LLM.

    Uses a checklist-based approach where each criterion is evaluated separately
    and scored from 0 to 100. The overall score is calculated as a weighted
    average of criterion satisfaction scores.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        # Parse config into Pydantic model for validation
        try:
            self._config_model = LLMJudgeConfig(**config)
        except Exception as e:
            logger.error(f"Invalid LLM judge config: {e}")
            raise ValueError(f"Invalid LLM judge configuration: {e}") from e

        self._llm: Optional[BaseChatModel] = None

    @property
    def config_model(self) -> LLMJudgeConfig:
        """Get the validated configuration model."""
        return self._config_model

    def _get_llm(self) -> BaseChatModel:
        """Lazy initialization of the judge LLM."""
        if self._llm is None:
            try:
                # Build LLM parameters with API key from environment
                llm_params: Dict[str, Any] = {
                    "temperature": self.config_model.temperature,
                }

                # Add API key based on provider
                if self.config_model.provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        llm_params["api_key"] = api_key
                    else:
                        logger.warning(
                            "OPENAI_API_KEY not found in environment. "
                            "LLM initialization may fail."
                        )
                elif self.config_model.provider == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if api_key:
                        llm_params["api_key"] = api_key
                    else:
                        logger.warning(
                            "ANTHROPIC_API_KEY not found in environment. "
                            "LLM initialization may fail."
                        )

                self._llm = init_chat_model(
                    model=self.config_model.model,
                    model_provider=self.config_model.provider,
                    **llm_params,
                )
                logger.info(
                    f"Judge LLM initialized: {self.config_model.model} "
                    f"({self.config_model.provider})"
                )
            except Exception as e:
                logger.error(f"Failed to initialize judge LLM: {e}")
                raise RuntimeError(f"Failed to initialize judge LLM: {e}") from e
        return self._llm

    def _parse_llm_response(
        self, response_text: str, criteria: List[Criterion]
    ) -> List[CriterionResult]:
        """
        Parse LLM response into criterion results.

        Args:
            response_text: Raw response from LLM
            criteria: List of criteria that were evaluated

        Returns:
            List of CriterionResult objects
        """
        # Clean up response text
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            judgment = json.loads(response_text)
            criterion_results_raw = judgment.get("criterion_results", [])

            # Validate and convert to CriterionResult objects
            criterion_results: List[CriterionResult] = []
            criterion_names = {c.name for c in criteria}

            for result_raw in criterion_results_raw:
                criterion_name = result_raw.get("criterion_name", "")
                if criterion_name not in criterion_names:
                    logger.warning(
                        f"Unknown criterion '{criterion_name}' in LLM response, skipping"
                    )
                    continue

                satisfaction_score = float(result_raw.get("satisfaction_score", 0.0))
                reasoning = result_raw.get("reasoning", "")

                # Get threshold from criterion if available, otherwise use default
                criterion = next(
                    (c for c in criteria if c.name == criterion_name), None
                )
                threshold = 70.0  # Default threshold
                if criterion:
                    # Could add per-criterion thresholds in the future
                    pass

                criterion_result = CriterionResult(
                    criterion_name=criterion_name,
                    satisfaction_score=satisfaction_score,
                    reasoning=reasoning,
                    passed=satisfaction_score >= threshold,
                )
                criterion_results.append(criterion_result)

            # Ensure all criteria have results (fill missing ones with 0)
            for criterion in criteria:
                if not any(
                    cr.criterion_name == criterion.name for cr in criterion_results
                ):
                    logger.warning(
                        f"Missing result for criterion '{criterion.name}', defaulting to 0"
                    )
                    criterion_results.append(
                        CriterionResult(
                            criterion_name=criterion.name,
                            satisfaction_score=0.0,
                            reasoning="Criterion not evaluated by LLM",
                            passed=False,
                        )
                    )

            return criterion_results

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(
                f"Failed to parse LLM judge response: {e}. Response: {response_text}"
            )
            # Return default results for all criteria
            return [
                CriterionResult(
                    criterion_name=criterion.name,
                    satisfaction_score=0.0,
                    reasoning=f"Failed to parse LLM response: {e}",
                    passed=False,
                )
                for criterion in criteria
            ]

    def _calculate_overall_score(
        self, criterion_results: List[CriterionResult], criteria: List[Criterion]
    ) -> float:
        """
        Calculate overall score from criterion results using weighted average.

        Args:
            criterion_results: Results for each criterion
            criteria: Original criteria list (for weights)

        Returns:
            Overall score from 0 to 100
        """
        if not criterion_results:
            return 0.0

        # Create a map of criterion names to weights
        criterion_weights = {c.name: c.weight for c in criteria}

        total_weighted_score = 0.0
        total_weight = 0.0

        for result in criterion_results:
            weight = criterion_weights.get(result.criterion_name, 1.0)
            total_weighted_score += result.satisfaction_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight

    def evaluate(
        self,
        query: str,
        result: LangGraphResult,
        reference: Optional[str] = None,
    ) -> MetricResult:
        """Evaluate using LLM judge with checklist-based criteria."""
        # Use reference from config if not provided
        effective_reference = reference or self.config_model.reference

        try:
            llm = self._get_llm()

            # Build the evaluation messages using the template
            messages = build_llm_judge_messages(
                query=query,
                agent_response=result.answer_text,
                criteria=self.config_model.criteria,
                reference=effective_reference,
            )

            # Get LLM judgment
            response = llm.invoke(messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse criterion results
            criterion_results = self._parse_llm_response(
                response_text, self.config_model.criteria
            )

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                criterion_results, self.config_model.criteria
            )

            # Store criterion weights in details for reference
            criterion_weights = {c.name: c.weight for c in self.config_model.criteria}

            return MetricResult(
                metric_name=self.name,
                metric_type="llm_judge",
                overall_score=overall_score,
                passed=overall_score >= self.config_model.threshold,
                criterion_results=criterion_results,
                details={
                    "judge_model": self.config_model.model,
                    "judge_provider": self.config_model.provider,
                    "criterion_weights": criterion_weights,
                    "num_criteria": len(self.config_model.criteria),
                },
                error=None,
            )

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}", exc_info=True)
            # Return failed result with default criterion results
            default_criterion_results = [
                CriterionResult(
                    criterion_name=criterion.name,
                    satisfaction_score=0.0,
                    reasoning=f"Evaluation failed: {str(e)}",
                    passed=False,
                )
                for criterion in self.config_model.criteria
            ]

            return MetricResult(
                metric_name=self.name,
                metric_type="llm_judge",
                overall_score=0.0,
                passed=False,
                criterion_results=default_criterion_results,
                details={},
                error=str(e),
            )


# Registry of available metric types
METRIC_REGISTRY: Dict[str, type[BaseMetric]] = {
    "llm_judge": LLMJudgeMetric,
}


def create_metric(metric_config: Dict[str, Any]) -> BaseMetric:
    """
    Create a metric instance from configuration.

    Args:
        metric_config: Dictionary with 'type' and other metric-specific config

    Returns:
        BaseMetric instance

    Example:
        {
            "type": "llm_judge",
            "name": "quality_score",
            "criteria": [
                {
                    "name": "clarity",
                    "description": "Is the response clear and easy to understand?",
                    "weight": 1.0
                }
            ],
            "threshold": 70.0
        }
    """
    metric_type = metric_config.get("type")
    if not metric_type:
        raise ValueError("Metric config must include 'type' field")

    metric_class = METRIC_REGISTRY.get(metric_type)
    if not metric_class:
        raise ValueError(
            f"Unknown metric type: {metric_type}. Available: {list(METRIC_REGISTRY.keys())}"
        )

    name = metric_config.get("name", metric_type)
    return metric_class(name=name, config=metric_config)


def register_metric(metric_type: str, metric_class: type[BaseMetric]) -> None:
    """Register a custom metric type."""
    METRIC_REGISTRY[metric_type] = metric_class
