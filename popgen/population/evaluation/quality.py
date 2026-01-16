"""Behavior quality evaluator."""

from __future__ import annotations

from typing import Any

import litellm
from pydantic import BaseModel

from .base import BaseEvaluator, EvalResult, calculate_stats

# ---------------------------------- PROMPTS --------------------------------- #

BEHAVIOR_QUALITY_SYSTEM_PROMPT = """You are an expert in evaluating the quality of agent behavior descriptions.
You will be provided with a behavior description and an environment description.
Your task is to evaluate the quality of the behavior across three rubrics:

1. **Relevance** (0-5): Is the behavior relevant to the environment?
   - 0: Completely irrelevant, behavior has nothing to do with the environment or its objectives
   - 1: Mostly irrelevant, only tangentially related to the environment
   - 2: Somewhat relevant, addresses the environment but misses key aspects
   - 3: Moderately relevant, relates to the environment but could apply to many other contexts
   - 4: Highly relevant, clearly designed for this environment with appropriate context
   - 5: Perfectly relevant, deeply tied to the environment's unique characteristics and objectives

2. **Realism** (0-5): Is the behavior realistic and achievable?
   - 0: Physically impossible or violates fundamental constraints of the environment
   - 1: Highly unrealistic, would require capabilities far beyond what agents possess
   - 2: Questionable realism, might be achievable under ideal conditions only
   - 3: Moderately realistic, achievable but with significant challenges or assumptions
   - 4: Realistic, clearly achievable by a competent agent with reasonable effort
   - 5: Highly realistic, straightforward to execute with well-understood mechanics

3. **Specificity** (0-5): Is the behavior specific enough to be actionable?
   - 0: Extremely vague, no actionable guidance (e.g., "play well")
   - 1: Very generic, could mean almost anything (e.g., "be strategic")
   - 2: Somewhat vague, general direction but lacking concrete details
   - 3: Moderately specific, provides guidance but leaves room for interpretation
   - 4: Specific, clear behavior and conditions with minor ambiguities
   - 5: Highly specific, precise behavior, triggers, and success criteria that leave no ambiguity

Provide detailed reasoning for each score."""

BEHAVIOR_QUALITY_USER_PROMPT = """Environment Description:
{environment_description}

Behavior:
{behavior}

Please evaluate this behavior's quality across the three rubrics (relevance, realism, specificity)."""

# ---------------------------------------------------------------------------- #


class RubricScore(BaseModel):
    """Score for a single rubric."""

    score: int  # 0-5
    reasoning: str


class BehaviorQualityResult(BaseModel):
    """Result of evaluating a behavior for quality."""

    relevance: RubricScore
    realism: RubricScore
    specificity: RubricScore


class BehaviorQualityEvaluator(BaseEvaluator):
    """Evaluator for behavior quality across relevance, realism, and specificity."""

    instance_plot_metrics = [
        ("relevance", "histogram"),
        ("realism", "histogram"),
        ("specificity", "histogram"),
    ]
    aggregate_plot_metrics = ["relevance", "realism", "specificity"]
    key_plot_metrics = [("total", "Quality Score")]

    def __init__(self, judge_model: str, num_workers: int = 8):
        super().__init__(name="BehaviorQuality", num_workers=num_workers)
        self.judge_model = judge_model

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Compute quality metrics for a single behavior.

        Args:
            input_data: Dict with keys 'behavior' and 'environment_description'

        Returns:
            Dict with relevance, realism, specificity scores and reasoning
        """
        behavior = input_data["behavior"]
        environment_description = input_data["environment_description"]

        user_prompt = BEHAVIOR_QUALITY_USER_PROMPT.format(
            behavior=behavior,
            environment_description=environment_description,
        )

        try:
            response = litellm.completion(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": BEHAVIOR_QUALITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=BehaviorQualityResult,
            )
            result = BehaviorQualityResult.model_validate_json(response.choices[0].message.content)
            return {
                "behavior": behavior,
                "relevance": result.relevance.score,
                "relevance_reasoning": result.relevance.reasoning,
                "realism": result.realism.score,
                "realism_reasoning": result.realism.reasoning,
                "specificity": result.specificity.score,
                "specificity_reasoning": result.specificity.reasoning,
                "total": result.relevance.score + result.realism.score + result.specificity.score,
                "error": None,
            }
        except Exception as e:
            return {
                "behavior": behavior,
                "relevance": 0,
                "relevance_reasoning": "",
                "realism": 0,
                "realism_reasoning": "",
                "specificity": 0,
                "specificity_reasoning": "",
                "total": 0,
                "error": str(e),
            }

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate instance-level metrics into overall metrics."""
        valid_metrics = [m for m in instance_metrics if m["error"] is None]
        n_errors = len(instance_metrics) - len(valid_metrics)

        return {
            "relevance": calculate_stats([m["relevance"] for m in valid_metrics]),
            "realism": calculate_stats([m["realism"] for m in valid_metrics]),
            "specificity": calculate_stats([m["specificity"] for m in valid_metrics]),
            "total": calculate_stats([m["total"] for m in valid_metrics]),
            "n_errors": n_errors,
        }

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        """Evaluate a list of behaviors and rollouts."""
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "Behavior Quality LLM-as-Judge",
                "judge_model": self.judge_model,
            }
        )

        return super().evaluate(inputs, metadata=metadata)
