"""Behavior adherence evaluator."""

from __future__ import annotations

from typing import Any

import litellm
import numpy as np
from pqdm.threads import pqdm
from pydantic import BaseModel

from popgen.utils.config import LitellmConfig
from popgen.utils.logging import encode_images_to_base64, rollout_states_to_str, uniform_sample
from popgen.utils.parsing import parse_llm_response

from .base import BaseEvaluator, EvalResult, calculate_stats

# ---------------------------------- PROMPTS --------------------------------- #

BEHAVIOR_ADHERENCE_SYSTEM_PROMPT = """You are an expert in evaluating agent behavior.
You will be provided with a behavior description and rollouts from the agent (trajectory states and possibly images sequences).
Your task is to evaluate how well the ego agent followed the behavior description based on these input rollouts.
Focus on the ego agent only, forget everything else including its teammates and opponents!
Evaluate whether its actions align with the described behavior, do not consider other goals.
You can use the rollout states to illustrate points.

Provide a concise reasoning and feedback in your evaluation."""

BEHAVIOR_ADHERENCE_USER_PROMPT = """Behavior: {behavior}
Ego Agent Name: {ego_name}
Ego Agent Visual Description: {ego_visual_description}

"""

# ---------------------------------------------------------------------------- #


class BehaviorAdherenceResult(BaseModel):
    """Result of evaluating a trajectory if it aligns with a specific behavior."""

    evaluation: bool
    reasoning: str

    def __str__(self) -> str:
        return f"Evaluation: {self.evaluation}\nReasoning: {self.reasoning}"


class BehaviorAdherenceEvaluator(BaseEvaluator):
    """Evaluator for behavior adherence based on rollout trajectories."""

    instance_plot_metrics = [("adherence", "bar")]
    aggregate_plot_metrics = ["adherence", "adjusted_adherence"]
    key_plot_metrics = [("adherence", "Behavior Adherence")]

    def __init__(self, model_config: LitellmConfig, num_workers: int = 8):
        super().__init__(name="BehaviorAdherence", num_workers=num_workers)
        self.model_config = model_config

    def compute_instance_metric(self, input_data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Compute adherence metric for a behavior rollouts.

        Args:
            input_data: Dict with keys:
                - behavior: str
                - ego_name: str
                - ego_visual_description: str
                - rollout_states: List[Dict[str, Any]]
                - rollout_images: Optional[List[List[np.ndarray]]]

        Returns:
            Dict with adherence evaluation and reasoning
        """
        behavior = input_data["behavior"]
        ego_name = input_data["ego_name"]
        ego_visual_description = input_data["ego_visual_description"]

        rollouts = list(
            zip(
                input_data["rollout_states"],
                input_data.get("rollout_images", [None] * len(input_data["rollout_states"])),
            )
        )

        def _worker_evaluate_rollout(rollout):
            rollout_states, rollout_images = rollout
            return self.evaluate_single_rollout(
                behavior=behavior,
                ego_name=ego_name,
                ego_visual_description=ego_visual_description,
                rollout_states=rollout_states,
                rollout_images=rollout_images,
                **kwargs,
            )

        all_results = pqdm(rollouts, _worker_evaluate_rollout, n_jobs=min(self.num_workers, len(rollouts)), desc="Evaluating rollouts", disable=True)

        # Aggregate results across rollouts
        n_adherent = sum(1 for r in all_results if r.evaluation)
        n_errors = sum(1 for r in all_results if "[ERROR]" in r.reasoning)
        n_evaluated = len(all_results) - n_errors
        adherence = n_adherent / len(all_results)
        adjusted_adherence = n_adherent / n_evaluated if n_evaluated > 0 else 0.0

        return {
            "adherence": adherence,
            "adjusted_adherence": adjusted_adherence,
            "n_adherent": n_adherent,
            "n_evaluated": n_evaluated,
            "n_errors": n_errors,
            "reasoning": combine_reasoning([r.model_dump() for r in all_results]),
        }

    def evaluate_single_rollout(self, behavior: str, ego_name: str, ego_visual_description: str, rollout_states: dict[str, Any], rollout_images: list[np.ndarray] = None, **kwargs) -> dict[str, Any]:
        """Evaluate adherence for a single rollout."""
        if isinstance(rollout_states, str) and "[ERROR]" in rollout_states:
            return BehaviorAdherenceResult(evaluation=False, reasoning=rollout_states)

        if rollout_images is None:
            rollout_images = []

        # Only use 5 equally spaced images if there are more
        if len(rollout_images) > 5:
            rollout_images = uniform_sample(rollout_images, n_samples=5)
        # Only use 20 equally spaced states if there are more
        if len(rollout_states) > 20:
            rollout_states = uniform_sample(rollout_states, n_samples=20)

        user_prompt = BEHAVIOR_ADHERENCE_USER_PROMPT.format(
            behavior=behavior,
            ego_name=ego_name,
            ego_visual_description=ego_visual_description,
        )
        content = [{"type": "text", "text": user_prompt}]

        prefix = f"Rollout (showing {len(rollout_states)} sampled states and {len(rollout_images)} sampled images):"
        separator = "-" * 60
        content.append({"type": "text", "text": f"{prefix}\n{separator}"})

        # Add all images
        for encoded_string in encode_images_to_base64(rollout_images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                }
            )

        # Add all states
        content.append({"type": "text", "text": rollout_states_to_str(rollout_states)})
        content.append({"type": "text", "text": separator})

        try:
            response = litellm.completion(
                model=self.model_config.model_name,
                messages=[
                    {"role": "system", "content": BEHAVIOR_ADHERENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=BehaviorAdherenceResult,
                **self.model_config.get_completion_kwargs(overrides=kwargs),
            )
            result = parse_llm_response(response, BehaviorAdherenceResult)
            return result
        except Exception as e:
            return BehaviorAdherenceResult(evaluation=False, reasoning=f"[ERROR] Evaluation failed: {e}")

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """Aggregate instance-level metrics into overall metrics."""
        # Filter out exceptions from pqdm
        valid_metrics = [m for m in instance_metrics if isinstance(m, dict)]
        n_failed = len(instance_metrics) - len(valid_metrics)

        return {
            "adherence": calculate_stats([m["adherence"] for m in valid_metrics]),
            "adjusted_adherence": calculate_stats([m["adjusted_adherence"] for m in valid_metrics]),
            "n_adherent": calculate_stats([m["n_adherent"] for m in valid_metrics]),
            "n_evaluated": calculate_stats([m["n_evaluated"] for m in valid_metrics]),
            "n_errors": calculate_stats([m["n_errors"] for m in valid_metrics]),
            "n_failed_workers": n_failed,
        }

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        """Evaluate a list of behaviors and rollouts."""
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "Behavior Adherence VLM-as-Judge",
                "judge_model": self.model_config.model_name,
                "num_behaviors": len(inputs),
            }
        )

        return super().evaluate(inputs, metadata=metadata)


def combine_reasoning(instance_metrics: list[dict[str, Any]]) -> str:
    """Combine reasoning from all instance metrics into a single string."""
    reasonings = []
    for i, m in enumerate(instance_metrics):
        reasoning = m.get("reasoning", "No reasoning provided.")
        success = m.get("evaluation", False)
        reasonings.append(f"> [{success}] Rollout #{i + 1}:\n{reasoning}\n")
    return "Rollouts reasonings:\n" + "\n".join(reasonings)
