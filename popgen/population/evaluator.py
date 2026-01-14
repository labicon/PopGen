from typing import Any

import litellm
import numpy as np
from pydantic import BaseModel

from popgen.utils.logging import encode_images_to_base64, rollout_states_to_str

# ---------------------------------- PROMPTS --------------------------------- #

EVALUATION_SYSTEM_PROMPT__NO_IMAGE = """You are an expert in evaluating agent behaviors.
You will be provided with a behavior description, {n_rollouts} rollouts from the agent (trajectory states).
Your task is to evaluate how well the ego agent followed the behavior description based on these input rollouts.
Focus on the ego agent only, forget everything else including its teammates and opponents!
Evaluate whether its actions align with the described behavior, do not consider other goals.
Give as much details and feedback in your reasoning. You can use the rollout states to illustrate."""

USER_PROMPT_TEMPLATE__NO_IMAGE = """Behavior: {behavior}
Ego Agent Name: {ego_name}
Ego Agent Visual Description: {ego_visual_description}

Rollout States:
"""

EVALUATION_SYSTEM_PROMPT = """You are an expert in evaluating agent behavior.
You will be provided with a behavior description, {n_rollouts} rollouts from the agent (trajectory states and images sequences).
Your task is to evaluate how well the ego agent followed the behavior description based on these input rollouts.
Focus on the ego agent only, forget everything else including its teammates and opponents!
Evaluate whether its actions align with the described behavior, do not consider other goals.
Give as much details and feedback in your reasoning. You can use the images and rollout states to illustrate."""

USER_PROMPT_TEMPLATE = """Behavior: {behavior}
Ego Agent Name: {ego_name}
Ego Agent Visual Description: {ego_visual_description}

Rollout States and Images:
"""

# ---------------------------------------------------------------------------- #


class EvaluationResult(BaseModel):
    """Result of evaluating a trajectory for a specific behavior."""

    evaluation: bool
    reasoning: str

    def __str__(self) -> str:
        return f"Evaluation: {self.evaluation}\nReasoning: {self.reasoning}"


class BehaviorEvaluator:
    def __init__(self, model: str):
        self.model = model

    def evaluate_rollouts(
        self,
        behavior: str,
        ego_name: str,
        ego_visual_description: str,
        rollouts_states: list[dict[str, Any]],
        rollouts_images: list[list[np.ndarray]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate agent behavior based on trajectory data and sequential images.

        Args:
            behavior: String representation of the behavior.
            ego_visual_description: Visual description of the ego agent.
            rollouts_states: List of state dictionaries for each rollout.
            rollouts_images: List of lists of images (as numpy arrays) for each rollout.

        Returns:
            Evaluation containing 'evaluation' and 'reasoning'.
        """

        system_prompt_template = EVALUATION_SYSTEM_PROMPT
        user_prompt_template__NO_IMAGE = USER_PROMPT_TEMPLATE

        if rollouts_images is None:
            system_prompt_template = EVALUATION_SYSTEM_PROMPT__NO_IMAGE
            user_prompt_template__NO_IMAGE = USER_PROMPT_TEMPLATE__NO_IMAGE
            rollouts_images = [[] for _ in rollouts_states]

        user_prompt = user_prompt_template__NO_IMAGE.format(
            behavior=behavior,
            ego_name=ego_name,
            ego_visual_description=ego_visual_description,
        )
        content = [{"type": "text", "text": user_prompt}]

        for i, (states, imgs) in enumerate(zip(rollouts_states, rollouts_images)):
            prefix = f"Rollout {i + 1} (showing {len(states)} sampled states and {len(imgs)} sampled images):"
            separator = "-" * 60
            content.append({"type": "text", "text": f"{prefix}\n{separator}"})

            # Add all images
            for encoded_string in encode_images_to_base64(imgs):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                    }
                )

            # Add all states
            content.append({"type": "text", "text": rollout_states_to_str(states)})
            content.append({"type": "text", "text": separator})

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt_template.format(n_rollouts=len(rollouts_states)),
                    },
                    {"role": "user", "content": content},
                ],
                response_format=EvaluationResult,
                **kwargs,
            )
            return EvaluationResult.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return EvaluationResult(
                evaluation=False,
                reasoning=f"Evaluation failed: {e}",
            )
