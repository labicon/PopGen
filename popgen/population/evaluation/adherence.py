"""Behavior adherence evaluator."""

from __future__ import annotations

from typing import Any

import litellm
from pydantic import BaseModel

from popgen.utils.config import LitellmConfig
from popgen.utils.logging import encode_images_to_base64, rollout_states_to_str, uniform_sample
from popgen.utils.parsing import parse_llm_response

from .base import BaseEvaluator, EvalResult

# ---------------------------------- PROMPTS --------------------------------- #

BEHAVIOR_ADHERENCE_SYSTEM_PROMPT = """You are an expert in evaluating whether agent actions align with a given behavior description.

You will be provided with a behavior description and rollouts from the agent (trajectory states and possibly images sequences).
Your task is to evaluate how well the ego agent followed the behavior description based on these input rollouts.
Evaluate whether its actions align with the described behavior, do not consider other goals.
Focus on the ego agent only, forget everything else including its teammates and opponents!
You can use the rollout states to illustrate points.

The visual description of the ego agent will be provided in the environment description.
In the trajectory logs, the ego agent is named agent0.

# Environment Description
{environment_description}

# Output Format
Return a JSON object with the following fields:
- "evaluation": boolean indicating whether the rollout showcases the behavior.
- "reasoning": short and concise explanation justifying the evaluation.
- "feedback": concise actionable instruction on what to change in the policy to better align with the behavior next time (focus on the *what*, not the *why*).
"""

BEHAVIOR_ADHERENCE_USER_PROMPT = """Behavior:
{behavior}"""

# ---------------------------------------------------------------------------- #


class BehaviorAdherenceResult(BaseModel):
    """Result of evaluating a trajectory if it aligns with a specific behavior."""

    evaluation: bool
    reasoning: str
    feedback: str

    def __str__(self) -> str:
        return f"Evaluation: {self.evaluation}\nReasoning: {self.reasoning}\nFeedback: {self.feedback}"


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
                - environment_description: str
                - rollout_states: List[Dict[str, Any]]
                - rollout_images: Optional[List[List[np.ndarray]]]

        Returns:
            Dict with adherence evaluation and reasoning
        """
        try:
            behavior = input_data["behavior"]
            environment_description = input_data["environment_description"]
            rollout_states = input_data["rollout_states"]
            rollout_images = input_data.get("rollout_images", [None] * len(input_data["rollout_states"]))

            # Error in rollout -> skip evaluation and return 0 adherence with error message as reasoning
            if isinstance(rollout_states, str) and "[ERROR]" in rollout_states:
                return {"adherence": 0.0, "reasoning": rollout_states, "feedback": "Fix the implementation."}

            system_prompt = BEHAVIOR_ADHERENCE_SYSTEM_PROMPT.format(
                environment_description=environment_description,
            )
            user_prompt = BEHAVIOR_ADHERENCE_USER_PROMPT.format(
                behavior=behavior,
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

            response = litellm.completion(
                model=self.model_config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                response_format=BehaviorAdherenceResult,
                **self.model_config.get_completion_kwargs(overrides=kwargs),
            )
            eval_result = parse_llm_response(response, BehaviorAdherenceResult)
        except Exception as e:
            eval_result = BehaviorAdherenceResult(evaluation=False, reasoning=f"[ERROR] Evaluation failed: {e}", feedback="")

        result = eval_result.model_dump()
        return {
            "adherence": 1.0 if result["evaluation"] else 0.0,
            "reasoning": result["reasoning"],
            "feedback": result["feedback"],
        }

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """Aggregate instance-level metrics into overall metrics."""
        n_adherent = sum(r["adherence"] for r in instance_metrics)
        n_errors = sum(1 for r in instance_metrics if "[ERROR]" in r["reasoning"])
        n_evaluated = len(instance_metrics) - n_errors

        return {
            "adherence": n_adherent / len(instance_metrics),
            "adjusted_adherence": n_adherent / n_evaluated if n_evaluated > 0 else 0.0,
            "n_evaluated": n_evaluated,
            "n_errors": n_errors,
        }

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        """Evaluate a list of behaviors and rollouts."""
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "Behavior Adherence VLM-as-Judge",
                "judge_model": self.model_config.model_name,
                "num_rollouts": len(inputs),
            }
        )

        return super().evaluate(inputs, metadata=metadata)
