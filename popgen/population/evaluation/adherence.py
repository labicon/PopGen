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

BEHAVIOR_ADHERENCE_SYSTEM_PROMPT = """You are an expert evaluator of whether an agent's actions match a given behavior description.

You will be given:
- a behavior description
- agent rollouts (trajectory states and possibly image sequences)

Your task is to determine whether the **ego agent's actions** adhere to the behavior description.

Guidelines:
- Evaluate alignment with the described behavior only.
- Do **not** consider any other possible goals.
- Focus **only** on the ego agent.
- Consider interactions with other agents only when it makes sense regarding behavior following or if it affects the ego agent's ability to demonstrate the behavior.
- You may reference rollout states in your explanation.
- Use states as the primary source of truth and images as supporting evidence.
- Some rollouts may provide serialized states instead of raw observations. Serialized states are structured summaries
  derived from the environment state, such as positions, distances, zones, nearest targets, and other task-relevant
  features. When serialized states are present, interpret them literally and use them as the main basis for factual claims.

In rare cases, the agent may be unable to fully demonstrate the expected behavior because of environment limitations or teammate actions.
If the ego agent still makes a clear, reasonable, good-faith effort to follow the behavior, you may mark it as adherent.
When you do this, you must explicitly explain the constraint and justify the decision in the reasoning field, for example:

{{
    "evaluation": true,
    "reasoning": "The behavior could not be fully demonstrated because of environment limitations, but the agent made a good-faith effort to follow it within those constraints."
}}

The environment description will specify how to identify the ego agent visually.
In the trajectory logs, the ego agent appears as `ego_agent` under `ego_team`.

# Environment Description
{environment_description}

# Output Format
Return a JSON object with exactly these fields:
- "evaluation": a boolean indicating whether the rollout demonstrates the behavior
- "reasoning": a detailed explanation of why the rollout does or does not demonstrate the behavior,
  including the single most decisive reason for the verdict and specific actions or states when relevant
"""

BEHAVIOR_ADHERENCE_USER_PROMPT = """Behavior:
{behavior}"""

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

    def __init__(self, model_config: LitellmConfig, num_workers: int = 1):
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
            serialize_state_fn = input_data.get("serialize_state_fn")

            # Error in rollout -> skip evaluation and return 0 adherence with error message as reasoning
            if isinstance(rollout_states, str) and "[ERROR]" in rollout_states:
                return {"adherence": 0.0, "reasoning": rollout_states}

            system_prompt = BEHAVIOR_ADHERENCE_SYSTEM_PROMPT.format(
                environment_description=environment_description,
            )
            user_prompt = BEHAVIOR_ADHERENCE_USER_PROMPT.format(
                behavior=behavior,
            )
            content = [{"type": "text", "text": user_prompt}]

            state_desc = "serialized sampled states" if serialize_state_fn is not None else "sampled states"
            prefix = f"Rollout (showing {len(rollout_states)} {state_desc} and {len(rollout_images)} sampled images in chronological order; step numbers are meaningful):"
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
            content.append({"type": "text", "text": rollout_states_to_str(rollout_states, serialize_state_fn=serialize_state_fn)})
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
            eval_result = BehaviorAdherenceResult(evaluation=False, reasoning=f"[ERROR] Evaluation failed: {e}")

        result = eval_result.model_dump()
        return {
            "adherence": 1.0 if result["evaluation"] else 0.0,
            "reasoning": result["reasoning"],
        }

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """Aggregate instance-level metrics into overall metrics."""
        n_adherent = sum(r["adherence"] for r in instance_metrics)
        n_errors = sum(1 for r in instance_metrics if "[ERROR]" in r["reasoning"])
        n_instances = len(instance_metrics)
        n_evaluated = n_instances - n_errors

        return {
            "adherence": n_adherent / n_instances if n_instances > 0 else 0.0,
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
