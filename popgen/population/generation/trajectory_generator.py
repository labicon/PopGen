"""Generator for full action trajectories with reasoning."""

from typing import Literal

from pydantic import BaseModel, Field, create_model

from popgen.population.generation.base import GeneratorWithFeedback
from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import parse_llm_response

SYSTEM_PROMPT = """You are an expert AI agent specializing in planning action sequences for multi-agent coordination tasks.

Your task is to generate a complete action trajectory that aligns with the given behavior description.

{environment_description}

ACTION SPACE:
{action_description}

RESPONSE FORMAT:
You must respond with a structured JSON object containing:
- actions: A complete list of actions for the entire rollout (length: {rollout_length} steps)
- reasoning: A high-level, concise explanation of your strategic approach and why you chose this trajectory

{example}"""


def create_trajectory_output(
    action_type: Literal["Discrete", "Continuous"],
    rollout_length: int,
    action_description: str,
) -> type[BaseModel]:
    """Dynamically create a Pydantic model for trajectory output.

    Args:
        action_type: Type of action space - "Discrete" or "Continuous".
        rollout_length: Number of steps in the rollout.
        action_description: Description of the action space.

    Returns:
        A Pydantic model class for trajectory output.
    """
    if action_type == "Continuous":
        actions_field = (list[list[float]], Field(description=f"A list of {rollout_length} continuous actions for the entire rollout. Each action is a list of floats (e.g., [fx, fy])."))
    else:  # Discrete
        actions_field = (list[int], Field(description=f"A list of {rollout_length} discrete action indices for the entire rollout."))

    return create_model(
        "TrajectoryOutput",
        actions=actions_field,
        reasoning=(str, Field(description="High-level, concise reasoning for the strategic approach and trajectory choices.")),
        action_description=(str, Field(default=action_description, description="Description of the action space.")),
    )


class Trajectory(BaseModel):
    """A complete action trajectory with reasoning."""

    actions: list[list[float]] | list[int]
    reasoning: str


class TrajectoryGenerator(GeneratorWithFeedback):
    """Generator for complete action trajectories based on behavior descriptions.

    This generator uses an LLM to create a full action sequence for an entire rollout,
    along with high-level reasoning explaining the strategic approach.

    Example:
        >>> from popgen.population.generation import TrajectoryGenerator
        >>>
        >>> generator = TrajectoryGenerator(model="gpt-4")
        >>>
        >>> result = generator.generate(
        ...     behavior="Play aggressively and always try to shoot at goal",
        ...     environment_description="You are playing football in a 2v2 game.",
        ...     action_description="Actions are forces [fx, fy] for movement.",
        ...     action_type="Continuous",
        ...     rollout_length=100,
        ... )
        >>>
        >>> print(result.reasoning)
        >>> print(len(result.actions))  # 100
    """

    def __init__(self, model_config: LitellmConfig):
        super().__init__(model_config, generation_type="Trajectory")

    def generate(
        self,
        behavior: str,
        environment_description: str,
        action_description: str,
        action_type: Literal["Discrete", "Continuous"],
        rollout_length: int,
        initial_observation: dict,
        feedbacks: list | None = None,
        **kwargs,
    ) -> Trajectory:
        """Generate a complete action trajectory for the given behavior.

        Args:
            behavior: The behavior description to guide the trajectory.
            environment_description: Description of the environment and task.
            action_description: Description of what the action output represents.
            action_type: Type of action space - "Discrete" or "Continuous".
            rollout_length: Number of steps in the trajectory.
            initial_observation: Initial observation/state to start trajectory from.
            feedbacks: Optional feedbacks to refine generation.
            **kwargs: Additional arguments passed to litellm.completion().

        Returns:
            Trajectory containing the action sequence and reasoning.
        """
        # Create the trajectory output model dynamically
        trajectory_model = create_trajectory_output(action_type=action_type, rollout_length=rollout_length, action_description=action_description)

        # Generate examples based on action type
        if action_type == "Continuous":
            example = """Example: {"actions": [[1.0, 0.5], [0.8, 0.3], ...], "reasoning": "I will start by moving forward aggressively..."}"""
        else:  # Discrete
            example = """Example: {"actions": [2, 1, 3, ...], "reasoning": "I will start by moving forward aggressively..."}"""

        # Build system prompt
        system_prompt = SYSTEM_PROMPT.format(
            environment_description=environment_description,
            action_description=action_description,
            rollout_length=rollout_length,
            example=example,
        )

        # Build user prompt with initial observation
        user_prompt = f"""Behavior description:
{behavior}

INITIAL OBSERVATION:
"""
        for key, value in initial_observation.items():
            if isinstance(value, (list, tuple)):
                user_prompt += f"- {key}: {value}\n"
            else:
                user_prompt += f"- {key}: {value}\n"

        user_prompt += f"""
Generate a complete action trajectory for {rollout_length} steps that implements this behavior."""

        # Call LLM
        response = super().generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            feedbacks=feedbacks,
            response_format=trajectory_model,
            **kwargs,
        )

        # Parse and return
        parsed = parse_llm_response(response, trajectory_model)

        return Trajectory(actions=parsed.actions, reasoning=parsed.reasoning)
