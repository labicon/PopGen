"""LLM-based text diversity discriminator."""

from __future__ import annotations

from typing import Any

import litellm
from pydantic import BaseModel

from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import parse_llm_response

from .base import BaseEvaluator, EvalResult, calculate_stats

# ---------------------------------- PROMPTS --------------------------------- #

DIVERSITY_SYSTEM_PROMPT = """You are an expert in evaluating the diversity of texts.
You will be provided with a target text and a list of other texts to compare against.
Your task is to evaluate how diverse the target text is compared to the other texts.

Consider the following aspects when evaluating diversity:
- Content and meaning: Does the text convey different ideas or concepts?
- Approach and style: Does the text use different approaches, perspectives, or styles?
- Specific details: Are the specific elements and patterns meaningfully different?
- Focus and emphasis: Does the text prioritize different aspects or themes?
- Novelty: How unique is this text compared to the others?

**Diversity Score** (0-5):
- 0: Nearly identical or extremely similar to other texts, no meaningful differences
- 1: Very similar, only superficial or minor differences in wording
- 2: Somewhat similar, shares most elements with others but has some variation
- 3: Moderately diverse, has distinct elements but shares some approaches with others
- 4: Highly diverse, represents a clearly different approach with unique elements
- 5: Extremely diverse, completely unique approach unlike any of the other texts

Provide your diversity score as an integer from 0 to 5, along with detailed reasoning."""

DIVERSITY_USER_PROMPT = """Target Text:
{target_text}

Other Texts to Compare Against:
{other_texts}

Please evaluate how diverse the target text is compared to the other texts."""

# ---------------------------------------------------------------------------- #


class DiversityScore(BaseModel):
    """Result of evaluating diversity for a single text."""

    diversity_score: int  # 0-5
    reasoning: str


class DiversityDiscriminator(BaseEvaluator):
    """Evaluator for text diversity using LLM-as-judge.

    For each text in the input list, compares it against all other texts
    and asks an LLM to rate how diverse it is on a scale of 0-5.
    """

    instance_plot_metrics = [("diversity_score", "histogram")]
    aggregate_plot_metrics = ["diversity_score"]
    key_plot_metrics = [("diversity_score", "Diversity Score")]

    def __init__(self, model_config: LitellmConfig, num_workers: int = 8):
        """Initialize the diversity discriminator.

        Args:
            model_config: LiteLLM configuration for the judge model.
            num_workers: Number of parallel workers for evaluation.
        """
        super().__init__(name="DiversityDiscriminator", num_workers=num_workers)
        self.model_config = model_config

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Compute diversity metric for a single text against all others.

        Args:
            input_data: Dict with keys:
                - 'target_text': The text to evaluate
                - 'other_texts': List of other texts to compare against
                - 'target_index': Optional index of the target text in the original list

        Returns:
            Dict with diversity_score, reasoning, and metadata
        """
        target_text = input_data["target_text"]
        other_texts = input_data["other_texts"]
        target_index = input_data.get("target_index", None)

        # Format other texts as a numbered list
        other_texts_formatted = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(other_texts)])

        user_prompt = DIVERSITY_USER_PROMPT.format(
            target_text=target_text,
            other_texts=other_texts_formatted,
        )

        try:
            response = litellm.completion(
                model=self.model_config.model_name,
                messages=[
                    {"role": "system", "content": DIVERSITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=DiversityScore,
                **self.model_config.get_completion_kwargs(),
            )
            result = parse_llm_response(response, DiversityScore)

            return {
                "target_text": target_text,
                "target_index": target_index,
                "diversity_score": result.diversity_score,
                "reasoning": result.reasoning,
                "num_comparisons": len(other_texts),
                "error": None,
            }
        except Exception as e:
            return {
                "target_text": target_text,
                "target_index": target_index,
                "diversity_score": 0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "num_comparisons": len(other_texts),
                "error": str(e),
            }

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate instance-level metrics into overall metrics.

        Args:
            instance_metrics: List of dicts from compute_instance_metric

        Returns:
            Dict with aggregated statistics across all texts
        """
        valid_metrics = [m for m in instance_metrics if m["error"] is None]
        n_errors = len(instance_metrics) - len(valid_metrics)

        return {
            "diversity_score": calculate_stats([m["diversity_score"] for m in valid_metrics]),
            "num_comparisons": calculate_stats([m["num_comparisons"] for m in valid_metrics]),
            "n_texts": len(instance_metrics),
            "n_errors": n_errors,
        }

    def evaluate(self, inputs: list[str | dict[str, Any]], metadata: dict[str, Any] = None) -> EvalResult:
        """Evaluate diversity for a list of texts.

        For each text in the list, compares it against all other texts
        and computes a diversity score.

        Args:
            inputs: List of text strings or dicts with 'text' or 'behavior' key.
                   For N texts, will make N comparisons (each 1 vs n-1).
            metadata: Optional metadata about the evaluation

        Returns:
            EvalResult with instance metrics for each text and aggregated stats
        """
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "LLM-as-Judge Diversity Discriminator",
                "judge_model": self.model_config.model_name,
                "num_texts": len(inputs),
            }
        )

        # Extract text from inputs
        texts = []
        for item in inputs:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                # Support 'text', 'behavior', or 'description' keys for backward compatibility
                texts.append(item.get("text", item.get("behavior", item.get("description", str(item)))))
            else:
                texts.append(str(item))

        # Create evaluation tasks: for each text, compare against all others
        eval_inputs = []
        for i, target_text in enumerate(texts):
            other_texts = [t for j, t in enumerate(texts) if j != i]
            eval_inputs.append(
                {
                    "target_text": target_text,
                    "other_texts": other_texts,
                    "target_index": i,
                }
            )

        return super().evaluate(eval_inputs, metadata=metadata)
