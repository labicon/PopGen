"""Trajectory-based agent wrapper that replays pre-generated action sequences."""

import json

import torch
import torch.nn as nn


class ReplayAgent(nn.Module):
    """Agent wrapper that replays a pre-generated trajectory.

    This wrapper loads a trajectory from a JSON file and replays the actions
    sequentially during rollout. Unlike LLMAgent which generates actions
    on-the-fly, this wrapper uses pre-computed trajectories.

    Attributes:
        trajectory_path: Path to the JSON file containing the trajectory.
        actions: List of pre-generated actions to replay.
        reasoning: High-level reasoning for the trajectory.

    Example:
        >>> from popgen.agents import ReplayAgent
        >>>
        >>> agent = ReplayAgent(trajectory_path="outputs/behavior_1/trajectory.json")
        >>> obs = {"observation": torch.randn(1, 2, 20)}  # (batch, agents, obs_dim)
        >>> actions = agent(obs)  # Returns next action in sequence
    """

    def __init__(self, trajectory_path: str):
        """Initialize the trajectory agent wrapper.

        Args:
            trajectory_path: Path to JSON file containing the trajectory data.
                Expected format: {"actions": [...], "reasoning": "..."}
        """
        super().__init__()
        self.trajectory_path = trajectory_path

        # Load trajectory from file
        with open(trajectory_path) as f:
            data = json.load(f)

        self.actions = data["actions"]
        self.reasoning = data.get("reasoning", "")

        # Track current step
        self._step_counter = 0

    def forward(self, obs):
        """Return the next action in the pre-generated trajectory.

        Args:
            obs: Observation dictionary with shape (num_envs, num_agents, obs_dim).
                Note: Observations are ignored since actions are pre-generated.

        Returns:
            Action tensor of shape (num_envs, num_agents, action_dim).
        """
        # Extract number of agents from observation
        obs_tensor = obs[0] if obs.dim() > 1 else obs
        if "observation" in obs_tensor:
            obs_tensor = obs_tensor["observation"]
        num_agents = len(obs_tensor)

        # Get current action from trajectory, with wraparound
        current_step = self._step_counter % len(self.actions)
        action = self.actions[current_step]

        # Convert to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32)

        # Replicate for all agents
        acts = [action_tensor for _ in range(num_agents)]

        # Increment step counter
        self._step_counter += 1

        # Stack actions and add batch dimension
        action_tensor = torch.stack(acts, dim=0)
        if obs.dim() > 1:
            action_tensor = action_tensor.unsqueeze(0)

        return action_tensor

    def reset(self):
        """Reset the step counter to restart trajectory replay."""
        self._step_counter = 0
