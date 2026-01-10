from typing import Any, Dict, List, Union
from pydantic import BaseModel
import litellm
import base64
import os
import io
from PIL import Image

from popgen.population.generation import Behavior
from popgen.utils.visualization import encode_image, encode_gif


class BehaviorEvaluation(BaseModel):
    """Result of evaluating a behavior against a trajectory."""
    evaluation: bool
    reasoning: str

    def __str__(self) -> str:
        return f"Evaluation: {self.evaluation}\nReasoning: {self.reasoning}"


EVALUATION_SYSTEM_PROMPT = """You are an expert in evaluating agent behaviors.
You will be provided with a behavior description and trajectory data.
Your task is to evaluate how well the agent followed the behavior description based on the trajectory data."""

EVALUATION_WITH_IMAGES_SYSTEM_PROMPT = """You are an expert in evaluating agent behaviors using both trajectory data and visual information.
You will be provided with a behavior description, trajectory data, and a sequence of images from the rollout.
Your task is to evaluate how well the agent followed the behavior description based on the trajectory data and the provided images."""


class BehaviorEvaluator:
    def __init__(self, model: str = "openai/gpt-5"):
        self.model = model

    def evaluate_trajectory(
        self,
        trajectory: str,
        behavior: Behavior | Dict[str, str],
    ) -> BehaviorEvaluation:
        """
        Evaluate agent behavior based on trajectory data.

        Args:
            trajectory: String representation of the trajectory.
            behavior: Behavior object or dictionary with 'name' and 'description'.

        Returns:
            BehaviorEvaluation containing 'evaluation' and 'reasoning'.
        """
        if isinstance(behavior, dict):
            name = behavior.get("name", "Unknown")
            description = behavior.get("description", "")
        else:
            name = behavior.name
            description = behavior.description

        user_prompt = f"""Behavior Name: {name}
Behavior Description: {description}
Trajectory Data:
{trajectory}

Evaluate if the agent followed the behavior description based on the trajectory data."""

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=BehaviorEvaluation,
                )
                return BehaviorEvaluation.model_validate_json(response.choices[0].message.content)
            except Exception as e:
                print(f"Evaluation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return BehaviorEvaluation(
                        evaluation=False,
                        reasoning=f"Evaluation failed: {e}",
                    )

    def evaluate_trajectory_and_images(
        self,
        trajectory: str,
        image_paths: List[str],
        behavior: Behavior | Dict[str, str],
    ) -> BehaviorEvaluation:
        """
        Evaluate agent behavior based on trajectory data and sequential images.

        Args:
            trajectory: String representation of the trajectory.
            image_paths: List of paths to image files or folders containing images.
            behavior: Behavior object or dictionary with 'name' and 'description'.

        Returns:
            BehaviorEvaluation containing 'evaluation' and 'reasoning'.
        """
        encoded_images = []
        try:
            for image_path in image_paths:
                # if gif 
                if image_path.endswith(".gif"):
                    encoded = encode_gif(image_path)
                    encoded_images.extend(encoded)
                else:
                    encoded = encode_image(image_path)
                    encoded_images.append(encoded)
        except Exception as e:
            return BehaviorEvaluation(
                evaluation=False,
                reasoning=f"Failed to read images: {e}",
            )

        if isinstance(behavior, dict):
            name = behavior.get("name", "Unknown")
            description = behavior.get("description", "")
        else:
            name = behavior.name
            description = behavior.description

        user_prompt = f"""Behavior Name: {name}
Behavior Description: {description}
Trajectory Data:
{trajectory}

Evaluate how well the agent 0 followed the behavior description based on the trajectory data and the provided sequential images.
It is not important to evaluate agent 1's behavior. Focus on agent 0 only.
In the images, agent 0 is red and agent 1 is blue.
Provide your evaluation and reasoning."""

        content = [{"type": "text", "text": user_prompt}]
        for encoded_string in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
            })

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EVALUATION_WITH_IMAGES_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    response_format=BehaviorEvaluation,
                )
                return BehaviorEvaluation.model_validate_json(response.choices[0].message.content)
            except Exception as e:
                print(f"Evaluation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return BehaviorEvaluation(
                        evaluation=False,
                        reasoning=f"Evaluation failed: {e}",
                    )