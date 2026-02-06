"""LLM-based agent wrapper for generating actions using language models."""

import copy
from typing import Literal, Optional

import litellm
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, create_model

from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import parse_llm_response

# ---------------------------------- PROMPTS --------------------------------- #

SYSTEM_PROMPT = """You are an intelligent agent playing in a multi-agent environment.

{environment_description}

Your goal is to observe the current state and decide on the best action to take.{behavioral_guideline}

ACTION SPACE:
{action_description}
{trajectory_history_section}
RESPONSE FORMAT:
You must respond with a structured JSON object containing:
- actions: A list of {horizon_length} action(s) for the planning horizon.

{example}"""

# ---------------------------------------------------------------------------- #


def create_horizon_action_output(
    action_type: Literal["Discrete", "Continuous"],
    horizon_length: int,
    action_description: str,
) -> type[BaseModel]:
    """Dynamically create a Pydantic model for actions over a planning horizon.

    Args:
        action_type: Type of action space - "Discrete" or "Continuous".
        horizon_length: Number of steps in the planning horizon.
        action_description: Description of the action space.

    Returns:
        A Pydantic model class for horizon actions.
    """
    if action_type == "Continuous":
        actions_field = (list[list[float]], Field(description=f"A list of {horizon_length} continuous actions over the planning horizon. Each action is a list of floats (e.g., [fx, fy])."))
    else:  # Discrete
        actions_field = (list[int], Field(description=f"A list of {horizon_length} discrete action indices over the planning horizon."))

    return create_model("HorizonActionOutput", actions=actions_field, action_description=(str, Field(default=action_description, description="Description of the action space.")))


class LLMAgent(nn.Module):
    """
    Wrapper that uses an LLM to generate actions based on observations.

    This wrapper integrates a Large Language Model as an agent policy, where the LLM
    receives environment observations and generates actions based on the environment
    description, action space description, and optional behavior specification.

    The wrapper supports multi-step planning via the horizon_length parameter. When
    horizon_length > 1, the LLM generates a trajectory of actions which are cached
    and used over multiple forward() calls, reducing the number of LLM API calls.

    The wrapper can also maintain trajectory history, allowing the LLM to see its
    previous action trajectory when generating new ones. This helps maintain
    consistency and strategic coherence across planning horizons.

    Attributes:
        model_config: LiteLLM configuration for the LLM model.
        environment_description: Description of the environment and task.
        action_description: Description of what the action output represents.
        action_type: Type of action space - "Discrete" or "Continuous".
        behavior: Optional description of the desired playing style/behavior.
        horizon_length: Number of steps to plan ahead (default: 1).
        include_trajectory_history: Whether to include previous trajectory in context (default: False).

    Example:
        >>> from popgen.agents import LLMAgent
        >>> from popgen.utils.config import LitellmConfig
        >>>
        >>> # Create an LLM agent with single-step planning
        >>> env_desc = "You are playing football in a 2v2 game. Your goal is to score."
        >>> action_desc = "Actions are forces [fx, fy] where fx controls horizontal movement (-1=left, 1=right) and fy controls vertical movement (-1=down, 1=up)."
        >>> model_config = LitellmConfig(model_name="gpt-4", generation_kwargs={"temperature": 0.7, "max_tokens": 100})
        >>> agent = LLMAgent(
        ...     model_config=model_config,
        ...     environment_description=env_desc,
        ...     action_description=action_desc,
        ...     action_type="Continuous",
        ...     behavior="Play aggressively and always try to shoot at goal"
        ... )
        >>>
        >>> # Create an LLM agent with 5-step planning horizon
        >>> model_config = LitellmConfig(model_name="gpt-4", generation_kwargs={"temperature": 0.7})
        >>> agent_multistep = LLMAgent(
        ...     model_config=model_config,
        ...     environment_description=env_desc,
        ...     action_description=action_desc,
        ...     action_type="Continuous",
        ...     horizon_length=5  # Plans 5 steps ahead
        ... )
        >>>
        >>> # Create an LLM agent with trajectory history
        >>> agent_with_history = LLMAgent(
        ...     model_config=model_config,
        ...     environment_description=env_desc,
        ...     action_description=action_desc,
        ...     action_type="Continuous",
        ...     horizon_length=3,
        ...     include_trajectory_history=True  # Include previous trajectory in context
        ... )
        >>>
        >>> # Create a discrete action LLM agent
        >>> agent_discrete = LLMAgent(
        ...     model_config=model_config,
        ...     environment_description=env_desc,
        ...     action_description="Choose action index: 0=up, 1=down, 2=left, 3=right",
        ...     action_type="Discrete"
        ... )
        >>>
        >>> # Use in rollout
        >>> obs = {"observation": torch.randn(1, 2, 20)}  # (batch, agents, obs_dim)
        >>> actions = agent(obs)  # Returns (1, 2, 2) tensor
    """

    def __init__(
        self,
        model_config: LitellmConfig,
        environment_description: str,
        action_description: str,
        action_type: Literal["Discrete", "Continuous"],
        behavior: str | None = None,
        horizon_length: int = 1,
        include_trajectory_history: bool = False,
    ):
        """
        Initialize the LLM agent wrapper.

        Args:
            model_config: LiteLLM configuration for the LLM model.
            environment_description: Description of the environment, task, and observation format.
            action_description: Description of what the action output represents and how it affects the environment.
            action_type: Type of action space - "Discrete" or "Continuous".
            behavior: Optional behavior description to guide the agent's playing style.
            horizon_length: Number of steps to plan ahead. The LLM generates actions for this many steps at once. Default is 1 (single-step planning).
            include_trajectory_history: Whether to include the previous trajectory in the context when generating new ones. Default is False.
        """
        super().__init__()
        self.environment_description = environment_description
        self.behavior = behavior
        self.model_config = model_config
        self.action_description = action_description
        self.horizon_length = horizon_length
        self.include_trajectory_history = include_trajectory_history
        self.action_type = action_type

        # Create the horizon action output model dynamically
        self.horizon_action_model = create_horizon_action_output(action_type=action_type, horizon_length=horizon_length, action_description=action_description)

        # Trajectory caching for multi-step planning
        self._cached_trajectories = []
        self._previous_trajectories = []
        self._step_counter = 0

        # Build the system prompt once during initialization
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt that describes the task and environment."""
        # Build conditional sections
        behavioral_guideline = ""
        if self.behavior:
            behavioral_guideline = f"""

BEHAVIORAL GUIDELINE:
You should play according to the following behavioral style:
{self.behavior}

This guideline describes how you should approach the task, but you must adapt your actions
based on the current observations to be effective."""

        trajectory_history_section = ""
        if self.include_trajectory_history:
            trajectory_history_section = """
TRAJECTORY HISTORY:
You will be provided with your previous trajectory to help you plan consistently.
Use this history to learn from past actions and maintain strategic coherence.

"""

        # Generate examples based on action type and horizon length
        if self.action_type == "Continuous":
            if self.horizon_length == 1:
                example = """Example: {"actions": [[1.0, 0.5]]}"""
            else:
                example = f"""Example for {self.horizon_length} step(s): {{"actions": [[1.0, 0.5], [0.8, 0.3], [0.5, 0.0]]}}"""
        else:  # Discrete
            if self.horizon_length == 1:
                example = """Example: {"actions": [2]}"""
            else:
                example = f"""Example for {self.horizon_length} step(s): {{"actions": [2, 1, 3]}}"""

        return SYSTEM_PROMPT.format(
            environment_description=self.environment_description,
            behavioral_guideline=behavioral_guideline,
            action_description=self.action_description,
            trajectory_history_section=trajectory_history_section,
            horizon_length=self.horizon_length,
            example=example,
        )

    def _build_trajectory_history_text(self, agent_idx: int) -> str:
        """
        Build text representation of previous trajectory for a specific agent.

        Args:
            agent_idx: Index of the agent.

        Returns:
            Formatted string describing the previous trajectory.
        """
        if not self.include_trajectory_history or not self._previous_trajectories:
            return ""

        if agent_idx < len(self._previous_trajectories):
            agent_trajectory = self._previous_trajectories[agent_idx]
            return f"\nPREVIOUS TRAJECTORY:\n{agent_trajectory}\n"

        return ""

    def _build_user_prompt(self, obs_dict: dict, agent_idx: int = 0) -> str:
        """
        Build the user prompt with the current observation and optional trajectory history.

        Args:
            obs_dict: Dictionary containing parsed observation information.
            agent_idx: Index of the agent (used for trajectory history).

        Returns:
            The formatted user prompt.
        """
        # Add trajectory history if enabled
        prompt = self._build_trajectory_history_text(agent_idx)

        # Add current observation
        prompt += "CURRENT OBSERVATION:\n"

        for key, value in obs_dict.items():
            if isinstance(value, (list, tuple)):
                prompt += f"- {key}: {value}\n"
            elif isinstance(value, torch.Tensor):
                prompt += f"- {key}: {value.tolist()}\n"
            else:
                prompt += f"- {key}: {value}\n"

        prompt += f"\nBased on this observation, what actions for the next {self.horizon_length} step{'s' if self.horizon_length > 1 else ''} should you take?"
        prompt += "\nProvide your action as a list of force vector [[fx, fy]]."

        return prompt

    def _parse_observation(self, obs: torch.Tensor) -> dict:
        """
        Parse the raw observation tensor into a human-readable dictionary.

        This is a basic parser that converts the observation tensor into a dict.
        Override this method for environment-specific observation parsing.

        Args:
            obs: Raw observation tensor of shape (obs_dim,).

        Returns:
            Dictionary with parsed observation components.
        """
        # Basic parsing - convert tensor to list
        # Subclasses can override this for more sophisticated parsing
        return {
            "raw_observation": obs.tolist(),
            "observation_shape": list(obs.shape),
        }

    def forward(self, obs):
        """
        Forward pass that generates actions using the LLM.

        When horizon_length > 1, the LLM generates a trajectory of actions
        which are cached and used over multiple forward calls to reduce
        the number of LLM API calls.

        Args:
            obs: Observation dictionary with shape (num_envs, num_agents, obs_dim).

        Returns:
            Action tensor of shape (num_envs, num_agents, action_dim).
        """
        # Extract observation for single env, all agents
        # Assuming obs has structure: {"observation": tensor}
        import time

        obs_tensor = obs[0] if obs.dim() > 1 else obs  # Extract single env
        if "observation" in obs_tensor:
            obs_tensor = obs_tensor["observation"]
        num_agents = len(obs_tensor)

        # Check if we need to generate new trajectories
        need_regenerate = self._step_counter == 0 or self._step_counter >= self.horizon_length

        if need_regenerate:
            # Generate new trajectories for all agents
            self._cached_trajectories = []

            for agent_idx, agent_obs in enumerate(obs_tensor):
                # Parse observation into readable format
                obs_dict = self._parse_observation(agent_obs)
                obs_dict["agent_id"] = agent_idx

                # Build user prompt with current observation and trajectory history
                user_prompt = self._build_user_prompt(obs_dict, agent_idx)

                # Prepare messages for LLM
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Call LLM with structured output
                try:
                    response = litellm.completion(
                        model=self.model_config.model_name,
                        messages=messages,
                        response_format=self.horizon_action_model,
                        **self.model_config.get_completion_kwargs(),
                    )

                    # Parse the structured response
                    horizon_output = parse_llm_response(response, self.horizon_action_model)

                    # Extract actions from horizon output
                    trajectory = horizon_output.actions

                    # Validate trajectory length
                    if len(trajectory) != self.horizon_length:
                        print(f"Warning: Expected {self.horizon_length} actions, got {len(trajectory)}. Padding/truncating.")
                        # Pad with appropriate zeros based on action type
                        if self.action_type == "Continuous":
                            zero_action = [0.0, 0.0]
                        else:
                            zero_action = 0

                        while len(trajectory) < self.horizon_length:
                            trajectory.append(zero_action)
                        # Truncate if too long
                        trajectory = trajectory[: self.horizon_length]

                    self._cached_trajectories.append(trajectory)

                except Exception as e:
                    # Fallback to zero trajectory if LLM fails
                    print(f"LLM trajectory generation failed: {e}. Using zero trajectory.")
                    if self.action_type == "Continuous":
                        zero_trajectory = [[0.0, 0.0] for _ in range(self.horizon_length)]
                    else:
                        zero_trajectory = [0 for _ in range(self.horizon_length)]
                    self._cached_trajectories.append(zero_trajectory)

            # Reset step counter
            self._step_counter = 0

        # Extract current step's action from cached trajectories
        acts = []
        for agent_idx in range(num_agents):
            action = torch.tensor(self._cached_trajectories[agent_idx][self._step_counter], dtype=torch.float32)
            acts.append(action)

        # Increment step counter
        self._step_counter += 1

        # Save completed trajectory to history if we've reached the end
        if self.include_trajectory_history and self._step_counter >= self.horizon_length:
            self._previous_trajectories = copy.deepcopy(self._cached_trajectories)

        # Stack actions and add batch dimension
        action_tensor = torch.stack(acts, dim=0)  # (?, num_agents, action_dim)
        if obs.dim() > 1:
            action_tensor = action_tensor.unsqueeze(0)  # (1, num_agents, action_dim)

        return action_tensor
