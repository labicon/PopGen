"""LLM-based agent wrapper for generating actions using language models."""

import copy
import logging
from typing import Literal

import litellm
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, create_model, model_validator

from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import parse_llm_response

from .utils import extract_observation_and_action_key, flatten_obs, format_action_output

LOGGER = logging.getLogger(__name__)


def _make_horizon_action_model(action_type: type[BaseModel], horizon_length: int) -> type[BaseModel]:
    """Build HorizonActionOutput that coerces raw scalar actions to action_type.

    LLMs sometimes return a list of raw scalars (e.g. [1, 1, 5]) instead of
    the expected list of structured objects (e.g. [{"action": 1}, ...]).
    The pre-validator wraps any non-dict item into {first_field: value} so
    Pydantic can validate it as the action_type.
    """
    first_field = next(iter(action_type.model_fields), None)

    @model_validator(mode="before")
    @classmethod
    def _coerce_actions(cls, data):
        if isinstance(data, dict) and "actions" in data and first_field is not None:
            coerced = []
            for item in data["actions"]:
                if not isinstance(item, (dict, action_type)):
                    item = {first_field: item}
                coerced.append(item)
            data = {**data, "actions": coerced}
        return data

    return create_model(
        "HorizonActionOutput",
        actions=(list[action_type], Field(description=f"A list of {horizon_length} actions over the planning horizon.")),
        __validators__={"_coerce_actions": _coerce_actions},
    )


# ---------------------------------- PROMPTS --------------------------------- #

SYSTEM_PROMPT = """You are an intelligent agent playing in a multi-agent environment.
Your goal is to observe the current state and decide on the best action to take given a style of play (behavior).

# Environment
{environment_description}

# Inputs
You will be provided with:
- Your behavioral guideline, which describes the style of play you should adopt.
- Your current observation of the environment, which may include your own state and the state of other agents and entities.
- Optional: Your previous trajectory of actions (if trajectory history is enabled).

# Outputs
The actions return type is defined as follows:
{action_description}
You must respond with a structured JSON object containing:
- actions: A list of these action(s) for the planning horizon. You will be provided with the horizon length, which indicates how many steps you should plan for.

Each action should be appropriate for the current observation and consistent with your behavioral guideline."""

FEEDBACK_SECTION = """

# Feedback from Prior Attempts
You previously attempted to play with your behavioral guideline but did not fully adhere to it.
Below is feedback from each prior attempt, including a sample of the trajectory and the evaluator's assessment.
Use this feedback to correct your behavior in the current session.

{feedback_entries}"""

USER_TEMPLATE = """# Behavioral Guideline:
{behavioral_guideline}

# Current Observation:
{agent_observation}

{trajectory_history_section}

You must plan for {horizon_length} step(s) ahead. Provide your actions as a list matching the description of the action space."""

# ---------------------------------------------------------------------------- #


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
        behavior: str,
        model_config: LitellmConfig,
        environment_description: str,
        action_type: type[BaseModel],
        horizon_length: int = 1,
        include_trajectory_history: bool = False,
        script_other_agents: bool = False,
        num_actions: int = 6,
        feedbacks: list[tuple[str, str]] = None,
    ):
        """
        Initialize the LLM agent wrapper.

        Args:
            behavior: Optional behavior description to guide the agent's playing style.
            model_config: LiteLLM configuration for the LLM model.
            environment_description: Description of the environment, task, and observation format.
            action_description: Description of what the action output represents and how it affects the environment.
            action_type: Pydantic BaseModel class defining the schema for a single action.
                The horizon model (actions: list[action_type]) is built automatically.
            horizon_length: Number of steps to plan ahead. The LLM generates actions for this many steps at once. Default is 1 (single-step planning).
            include_trajectory_history: Whether to include the previous trajectory in the context when generating new ones. Default is False.
            script_other_agents: If True, non-ego agents (index > 0) take random actions instead of using the LLM. Default is False.
            num_actions: Size of the discrete action space for random scripted agents. Default is 6 (LBF action space).
        """
        super().__init__()
        self.behavior = behavior
        self.model_config = model_config
        self.environment_description = environment_description
        self.horizon_length = horizon_length
        self.include_trajectory_history = include_trajectory_history
        self.action_type = action_type
        self.script_other_agents = script_other_agents
        self.num_actions = num_actions

        # Build horizon action model: actions is a list of single-action models
        self.horizon_action_model = _make_horizon_action_model(action_type, horizon_length)

        # Trajectory caching for multi-step planning
        self._cached_trajectories = []
        self._previous_trajectories = []
        self._step_counter = 0

        # Build the system prompt once during initialization
        self.system_prompt = SYSTEM_PROMPT.format(
            environment_description=self.environment_description,
            action_description=self.action_type.__doc__,
        )
        if feedbacks:
            entries = []
            for n, (trajectory, reasoning) in enumerate(feedbacks, 1):
                entry = f"## Attempt #{n}\n"
                if trajectory:
                    entry += f"### Trajectory:\n{trajectory}\n\n"
                entry += f"### Evaluator Assessment:\n{reasoning}"
                entries.append(entry)
            self.system_prompt += FEEDBACK_SECTION.format(feedback_entries="\n\n".join(entries))

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
            return f"# Previous Trajectory:\n{agent_trajectory}"

        return ""

    # TODO: why?
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

    def _pad_trajectory(self, trajectory: list) -> list:
        """Pad or truncate the trajectory to match the horizon length."""
        if len(trajectory) < self.horizon_length:
            trajectory += [self.action_type.default_action().to_tensor()] * (self.horizon_length - len(trajectory))
        return trajectory[: self.horizon_length]

    def _act(self, obs, agent_idx: int):
        """Forward act on a single environment and agent observation without trajectory caching."""
        # Parse observation into readable format
        obs_dict = self._parse_observation(obs)
        obs_dict["agent_id"] = agent_idx

        # Build user prompt with current observation and trajectory history
        agent_observation = "\n".join([f"{key}: {value}" for key, value in obs_dict.items()])
        trajectory_history_section = self._build_trajectory_history_text(agent_idx)
        user_prompt = USER_TEMPLATE.format(
            behavioral_guideline=self.behavior,
            agent_observation=agent_observation,
            trajectory_history_section=trajectory_history_section,
            horizon_length=self.horizon_length,
        )

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

            # Parse the structured response and convert action models to raw values
            trajectory = [a.to_tensor() for a in parse_llm_response(response, self.horizon_action_model).actions]

            # Validate trajectory length and convert to tensor
            trajectory = self._pad_trajectory(trajectory)

        except Exception as e:
            # Fallback to zero trajectory if LLM fails
            LOGGER.warning(f"LLM trajectory generation failed: {e}. Using zero trajectory.")
            trajectory = [self.action_type.default_action().to_tensor() for _ in range(self.horizon_length)]

        trajectory = torch.stack(trajectory)  # Shape: (horizon_length, action_dim)

        return trajectory

    def forward(self, input_td):
        """
        Forward pass that generates actions using the LLM.

        When horizon_length > 1, the LLM generates a trajectory of actions
        which are cached and used over multiple forward calls to reduce
        the number of LLM API calls.

        Args:
            input_td: Input observation container (TensorDict preferred).

        Returns:
            TensorDict with action when input is TensorDict, otherwise action tensor.
        """
        obs_tensor, action_key = extract_observation_and_action_key(input_td)
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(f"LLMAgent expected a torch.Tensor observation, got {type(obs_tensor).__name__}.")

        if obs_tensor.dim() == 0:
            raise ValueError("LLMAgent expects observation tensor with at least 1 dimension.")

        flat_obs, leading_shape = flatten_obs(obs_tensor)

        n_obs_slots = flat_obs.shape[0]

        # Check if we need to generate new trajectories
        if self._step_counter == 0 or self._step_counter >= self.horizon_length or len(self._cached_trajectories) != n_obs_slots:
            # Generate new trajectories for all agents
            self._step_counter = 0
            self._cached_trajectories = []

            for agent_idx, agent_obs in enumerate(flat_obs):
                trajectory = self._act(agent_obs, agent_idx)
                self._cached_trajectories.append(trajectory)

        # Extract current step's action from cached trajectories
        action_flat = torch.stack(self._cached_trajectories)[:, self._step_counter]
        if len(leading_shape) == 0:
            action_tensor = action_flat[0]
        else:
            action_tensor = action_flat.reshape(*leading_shape, *action_flat.shape[1:])

        # Increment step counter
        self._step_counter += 1

        # Save completed trajectory to history if we've reached the end
        if self.include_trajectory_history and self._step_counter >= self.horizon_length:
            self._previous_trajectories = copy.deepcopy(self._cached_trajectories)

        return format_action_output(input_td, action_key, action_tensor)
