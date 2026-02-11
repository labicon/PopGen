from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym

# Ensure lbforaging is registered
import lbforaging  # noqa: F401
import numpy as np
import torch
from pydantic import BaseModel, Field

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
def reward_fn(self, agent: Player):
    # You are designing reward for 'agent' in the environment.
    # self is the environment instance (ForagingEnv).
    # self.players gives list of all players.
    # self.field gives grid status.

    # Calculate custom reward
    reward = 0.0

    y, x = agent.position
    min_dist = float('inf')
    rows, cols = self.field.shape
    for r in range(rows):
        for c in range(cols):
            if self.field[r, c] > 0: # If there is food
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
        """Convert action to a torch tensor of shape (1,)."""
        return torch.tensor([self.action], dtype=torch.int64)


# ---------------------------------------------------------------------------- #
#                              State Extraction                                #
# ---------------------------------------------------------------------------- #


def get_state_lbf(env) -> dict[str, Any]:
    """
    Extracts the global state from the LBF environment.
    """
    state = {}

    # Access unwrapped env for internal state
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    # Use nested 'agents' dict to mimic VMAS structure for compatibility with logging
    # Food is also added here so it gets printed by rollout_states_to_str
    agents_data = {}

    # Agents
    # env.players is list of Player objects
    for i, player in enumerate(env.players):
        agents_data[f"agent{i}"] = {
            "pos": [int(x) for x in player.position],
            "level": int(player.level),
            "reward": float(player.reward) if isinstance(player.reward, (int, float, np.number)) else player.reward,
            "score": float(player.score) if isinstance(player.score, (int, float, np.number)) else player.score,
        }

    # Food
    # env.field is ndarray of shape (rows, cols) from env
    rows, cols = env.field.shape
    food_list = []
    for r in range(rows):
        for c in range(cols):
            if env.field[r, c] > 0:
                food_list.append(((r, c), env.field[r, c]))

    # Sort food to have consistent order
    food_list.sort(key=lambda x: x[0])

    for i, (pos, level) in enumerate(food_list):
        agents_data[f"food{i}"] = {"pos": [int(x) for x in pos], "level": int(level)}

    state["agents"] = agents_data
    return state


# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy.
    """
    # if env is not defined, load default lbf environment
    if env is None:
        # Foraging-8x8-2p-2f-v3
        env = gym.make("Foraging-8x8-2p-2f-v3", render_mode="rgb_array")

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 50

    return rollout_lbf_env(policy=policy, env=env, get_state_fn=get_state_lbf, **kwargs)


def train_fn(reward_fn: Callable, output_dir: str, **kwargs):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward (float or tensor).
        output_dir: Directory to save results.
    """
    return train_lbf_env(reward_fn=reward_fn, output_dir=output_dir, rl_cfg_dir=RL_CFG_DIR, **kwargs)


# ---------------------------------------------------------------------------- #
#                              Debug / Testing                                 #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    import torch
    from lbforaging.agents.heuristic_agent import H1
    from tensordict import TensorDict

    from popgen.utils.logging import save_rollout

    class H1Policy(torch.nn.Module):
        def __init__(self, env):
            super().__init__()
            self.env = env
            self.agents = [H1(p) for p in env.unwrapped.players]
            self.dummy_param = torch.nn.Parameter(torch.empty(0))

        def forward(self, batch):
            # We ignore batch observations and use internal env state
            # This is strictly for evaluation of heuristic agents in this harness
            actions = []
            for agent in self.agents:
                obs = self.env.unwrapped._make_obs(agent.player)
                action = agent._step(obs)
                actions.append(action.value)

            return TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "action": torch.tensor(actions).unsqueeze(0)  # (1, n_agents)
                        },
                        batch_size=[1, len(self.agents)],
                    )
                },
                batch_size=[1],
            )

    # Create simplified env
    # Foraging-8x8-2p-2f-coop-v3
    env_id = "Foraging-8x8-2p-2f-coop-v3"
    env = gym.make(env_id, render_mode="rgb_array")

    print(f"Created env: {env_id}")

    policy = H1Policy(env)

    print("Starting rollout...")
    states, images, _ = rollout_lbf_env(policy=policy, env=env, get_state_fn=get_state_lbf, max_steps=50)

    print(f"Rollout finished. Steps: {len(states)}")
    print("Example state:", states[0])

    save_path = Path("./lbf_rollout_output")
    print(f"Saving rollout to {save_path}...")
    save_rollout(states, images, save_path)
    print("Rollout saved.")
