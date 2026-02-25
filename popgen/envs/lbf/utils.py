import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

# Ensure lbforaging is registered
import lbforaging  # noqa: F401
import numpy as np
import torch
from benchmarl.environments.lbf.common import LbfTask
from pydantic import BaseModel, Field

from popgen.agents.utils import extract_observation_and_action_key, flatten_obs, format_action_output
from popgen.envs.lbf_common import rollout_lbf_env, train_lbf_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

ENVIRONMENT_DESCRIPTION = """The environment is called "Level-Based Foraging" (LBF).
It is a 8x8 grid world where a team of agents must coordinate to collect foods.

Each food has a specific level (difficulty to carry).
Each agent has a specific level (carrying capability).
Agents can move in 4 directions or load the food.
If the food level is higher than the agent level, multiple agents must load the food together (cooperative transport).
The sum of levels of valid agents loading the food must be greater than or equal to the food level.
If multiple agents are loading the same food, they must load at the same time step.

Some visual aspects:
- Ego agent is the blue agent. The white one is its teammate.
- The number associated top each agent and food represent its level.

Any food or agent cannot be on the same cell as another food or agent.
They can load the food only when they are on the neighboring (up, down, left, or right) cell of the food (not on the food cell).

> **Observation Space**:
The observation is a flattened vector that includes the positions and levels of all agents and foods in the grid.
The order of the vector is as follows:
- For each food (num_foods*3 elements): [Food_row, Food_col, Food_level] – The row and column indices of the food in the grid, and its level.
- For each agent (num_agents*3 elements): [Agent_row, Agent_col, Agent_level] – The row and column indices of the agent in the grid, and its level.

Note that row index `r` increases as you go down, and column index `c` increases as you go right.
Both indices start from 0 at the top-left corner.

> **Action Space**:
The action space is discrete with the following mapping:
0: NONE
1: NORTH
2: SOUTH
3: WEST
4: EAST
5: LOAD

For example to move NORTH, you would return 1."""

REWARD_EXAMPLE = """
Behavior: agent stays close to food
```python
def reward_fn(env, agent):
    # You are designing reward for 'agent' in the environment.
    # env is the environment instance (ForagingEnv).
    # env.players gives list of all players.
    # env.field gives grid status.

    # Calculate custom reward
    reward = 0.0

    y, x = agent.position
    min_dist = float('inf')
    rows, cols = env.field.shape
    for r in range(rows):
        for c in range(cols):
            if env.field[r, c] > 0: # If there is food
                dist = abs(r - y) + abs(c - x)
                if dist < min_dist:
                    min_dist = dist

    if min_dist < float('inf'):
       reward += 1.0 / (min_dist + 1)

    return reward
```
"""

# ---------------------------------------------------------------------------- #
#                                 Action Type                                  #
# ---------------------------------------------------------------------------- #


class ActionType(BaseModel):
    """
    Action type for the LBF environment.
    - action: is an integer representing the action to take in the LBF environment.

    The mapping of integer to action is:
    0: NONE
    1: NORTH
    2: SOUTH
    3: WEST
    4: EAST
    5: LOAD

    Example: to move NORTH, you would return ActionType(action=1).
    """

    action: int = Field(..., description="Integer representing the action to take in the LBF environment.")

    @classmethod
    def default_action(cls):
        """Default action (NONE)."""
        return cls(action=0)

    def to_tensor(self):
        """Convert action to a scalar torch tensor."""
        return torch.tensor(self.action, dtype=torch.int64)


# ---------------------------------------------------------------------------- #
#                              State Extraction                                #
# ---------------------------------------------------------------------------- #


def get_state_lbf(env) -> dict[str, Any]:
    """
    Extracts the global state from the LBF environment.
    Returns {"agents": {agent_name: {...}}, "env": {food info}}.
    """
    # Access unwrapped env for internal state
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    # Agents
    agents_data = {}
    for i, player in enumerate(env.players):
        agents_data[f"agent{i}"] = {
            "pos": [int(x) for x in player.position],
            "level": int(player.level),
            "reward": float(player.reward) if isinstance(player.reward, (int, float, np.number)) else player.reward,
            "score": float(player.score) if isinstance(player.score, (int, float, np.number)) else player.score,
        }

    # Food (env-level state)
    rows, cols = env.field.shape
    food_list = []
    for r in range(rows):
        for c in range(cols):
            if env.field[r, c] > 0:
                food_list.append(((r, c), env.field[r, c]))
    food_list.sort(key=lambda x: x[0])

    env_data = {}
    for i, (pos, level) in enumerate(food_list):
        env_data[f"food{i}_pos"] = [int(x) for x in pos]
        env_data[f"food{i}_level"] = int(level)

    return {"agents": agents_data, "env": env_data}


# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #


def _extract_env_metadata(env) -> tuple[int, int]:
    num_agents = int(env.num_agents)
    # Obs dim can be from "agents" "observation" or "ego" "observation" depending on the env wrapper, but they should have the same shape
    try:
        obs_dim = env.observation_spec["ego", "observation"].shape[-1]
    except KeyError:
        obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    num_foods = int((obs_dim - 3 * num_agents) // 3)
    return num_agents, num_foods


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy.
    """
    # if env is not defined, load default lbf environment
    if env is None:
        # Foraging-8x8-2p-2f-v3
        env = create_env(**kwargs)

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 50

    # If policy is a list of policies, wrap it into a MultiPolicyWrapper with ego and teammate roles
    if isinstance(policy, list) and len(policy) > 1:
        from popgen.agents.scripted_partner import MultiPolicyWrapper

        policy = MultiPolicyWrapper(policies=policy)

    return rollout_lbf_env(
        policy=policy,
        env=env,
        get_state_fn=get_state_lbf,
        **kwargs,
    )


def train_fn(reward_fn: Callable, output_dir: str, **kwargs):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward (float or tensor).
        output_dir: Directory to save results.
    """
    scripted_policy = kwargs.pop("scripted_policy", None)
    if scripted_policy is None:
        temp_env = create_env(num_envs=1, seed=0, device="cpu")
        try:
            scripted_policy = [ScriptedPolicy(temp_env)]
        finally:
            temp_env.close()

    return train_lbf_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        rl_cfg_dir=RL_CFG_DIR,
        teammate_policy=scripted_policy,
        **kwargs,
    )

# ---------------------------------------------------------------------------- #
#                              Scripted Policy                                 #
# ---------------------------------------------------------------------------- #


class ScriptedPolicy(torch.nn.Module):
    """Heuristic policy that moves toward the closest food and loads when adjacent.

    Parses the flattened LBF observation vector to extract food/agent positions,
    finds the nearest food by Manhattan distance, moves toward it, and issues
    LOAD when adjacent (distance == 1).

    Args:
        env: BenchMARL LBF environment created by create_env.
    """

    # Action constants
    NONE = 0
    NORTH = 1  # row - 1
    SOUTH = 2  # row + 1
    WEST = 3   # col - 1
    EAST = 4   # col + 1
    LOAD = 5

    def __init__(self, env):
        super().__init__()
        self.num_agents, self.num_foods = _extract_env_metadata(env)
        # nn.Module requires at least one parameter for some codepaths
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action for each observation in the batch.

        Args:
            obs: Observation tensor, shape ``(..., obs_dim)``.

        Returns:
            Action tensor, shape ``(...)`` with integer action indices.
        """
        obs_tensor, action_key = extract_observation_and_action_key(obs)
        flat_obs, leading_shape = flatten_obs(obs_tensor)
        batch_size = flat_obs.shape[0]

        actions = torch.zeros(batch_size, dtype=torch.long, device=obs_tensor.device)

        for b in range(batch_size):
            o = flat_obs[b]

            # Self position (first agent entry, after all food entries)
            agent_offset = self.num_foods * 3
            self_row = o[agent_offset].item()
            self_col = o[agent_offset + 1].item()

            # Find closest visible food
            min_dist = float("inf")
            closest_food_row, closest_food_col = -1.0, -1.0
            for i in range(self.num_foods):
                food_row = o[3 * i].item()
                food_col = o[3 * i + 1].item()
                food_level = o[3 * i + 2].item()
                if food_row < 0 or food_level <= 0:
                    continue
                dist = abs(food_row - self_row) + abs(food_col - self_col)
                if dist < min_dist:
                    min_dist = dist
                    closest_food_row, closest_food_col = food_row, food_col

            if min_dist == float("inf"):
                actions[b] = self.NONE
                continue

            # Adjacent to food -> LOAD
            if min_dist <= 1:
                actions[b] = self.LOAD
                continue

            # Move toward the closest food (prefer larger axis difference)
            row_diff = closest_food_row - self_row
            col_diff = closest_food_col - self_col

            if abs(row_diff) >= abs(col_diff):
                actions[b] = self.NORTH if row_diff < 0 else self.SOUTH
            else:
                actions[b] = self.WEST if col_diff < 0 else self.EAST

        if len(leading_shape) == 0:
            actions = actions.squeeze(0)
        else:
            actions = actions.reshape(*leading_shape)
        return format_action_output(obs, action_key, actions)


# ---------------------------------------------------------------------------- #
#                              Make env function                               #
# ---------------------------------------------------------------------------- #


def create_env(num_envs=1, continuous_actions=True, seed=None, device="cpu", **kwargs):
    """
    Creates the LBF environment using BenchMARL's LbfTask.

    Args:
        kwargs: Keyword arguments to pass to the environment creation function.
            Can include num_envs, device, continuous_actions, seed, etc.

    Returns:
        An instance of the LBF environment.
    """
    return LbfTask.LBF.get_from_yaml(path=str(RL_CFG_DIR / "lbf.yaml")).get_env_fun(
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
    )()

