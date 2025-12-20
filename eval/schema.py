"""
Pydantic models for evaluation metrics.

This module defines the data structures used throughout the evaluation system,
following clean architecture principles with clear separation of concerns.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Criterion(BaseModel):
    """A single evaluation criterion in a checklist."""

    name: str = Field(..., description="Name/identifier of the criterion")
    description: str = Field(..., description="Description of what to evaluate")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight of this criterion in overall score calculation",
    )

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure weight is positive."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v


class CriterionResult(BaseModel):
    """Result of evaluating a single criterion."""

    criterion_name: str = Field(..., description="Name of the criterion evaluated")
    satisfaction_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Satisfaction score from 0 to 100",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the evaluation for this criterion",
    )
    passed: bool = Field(
        default=False,
        description="Whether this criterion passed (satisfaction_score >= threshold)",
    )

    @field_validator("satisfaction_score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to [0, 100] range."""
        return max(0.0, min(100.0, v))


class LLMJudgeConfig(BaseModel):
    """Configuration for LLM-as-judge metric."""

    name: str = Field(..., description="Name of the metric")
    criteria: List[Criterion] = Field(
        ...,
        min_length=1,
        description="List of criteria to evaluate (checklist)",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for judging",
    )
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, etc.)",
    )
    threshold: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Minimum overall score (0-100) to pass",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM (0.0 for deterministic judging)",
    )
    reference: Optional[str] = Field(
        default=None,
        description="Optional reference answer for comparison",
    )


class MetricResult(BaseModel):
    """Result from evaluating a metric."""

    metric_name: str = Field(..., description="Name of the metric")
    metric_type: str = Field(..., description="Type of metric (e.g., 'llm_judge')")
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall score from 0 to 100",
    )
    passed: bool = Field(
        ...,
        description="Whether the metric passed (overall_score >= threshold)",
    )
    criterion_results: List[CriterionResult] = Field(
        default_factory=list,
        description="Results for each criterion in the checklist",
    )
    details: dict = Field(
        default_factory=dict,
        description="Additional details about the evaluation",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if evaluation failed",
    )

    @field_validator("overall_score")
    @classmethod
    def clamp_overall_score(cls, v: float) -> float:
        """Clamp overall score to [0, 100] range."""
        return max(0.0, min(100.0, v))

    def calculate_overall_score(self) -> float:
        """
        Calculate overall score from criterion results using weighted average.

        Returns:
            Overall score from 0 to 100
        """
        if not self.criterion_results:
            return 0.0

        total_weighted_score = 0.0
        total_weight = 0.0

        # Get weights from criteria (we need to match by name)
        # For now, assume equal weights if not provided in details
        criterion_weights = self.details.get("criterion_weights", {})

        for result in self.criterion_results:
            weight = criterion_weights.get(result.criterion_name, 1.0)
            total_weighted_score += result.satisfaction_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight
