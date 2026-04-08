import inspect
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure lbforaging is registered
import lbforaging  # noqa: F401
import torch
import torch.nn as nn
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments.lbf.common import LbfGymWrapper, LbfTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from pydantic import BaseModel, Field
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, TransformedEnv

from popgen.envs.utils import apply_rollout_seed

# Enable headless rendering for pyglet (used by lbforaging)
os.environ.setdefault("PYGLET_HEADLESS", "true")

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

ENVIRONMENT_DESCRIPTION = """
Environment: Level-Based Foraging (LBF)

This environment is an 8x8 grid world where a team of agents must coordinate to collect food items.

--------------------------------------------------
Grid and Entities
--------------------------------------------------

- The grid size is 8 × 8.
- Multiple agents cooperate to collect food.
- Each agent and food occupies a unique grid cell (no overlaps).

Each entity has a **level**:

- Agent level: carrying capability
- Food level: difficulty required to carry the food

Food can only be collected if the **sum of levels of all agents loading it is greater than or equal to the food level**.

--------------------------------------------------
Cooperative Loading
--------------------------------------------------

Agents can attempt to load food using the LOAD action.

Rules:

- An agent must be **cardinally adjacent** to the food: either at (row ± 1, col) or (row, col ± 1).
- Diagonal positions are **not** adjacent.
- Agents **cannot stand on the food cell**.

If a food's level exceeds a single agent's level:

- Multiple agents must **LOAD at the same time step**.
- The **sum of their levels** must meet or exceed the food level.

--------------------------------------------------
Visual Representation
--------------------------------------------------

- The ego agent is shown in **blue**.
- The teammate agent is **black**.
- Numbers displayed next to agents and food indicate their **levels**.

--------------------------------------------------
Observation Space
--------------------------------------------------

The observation is a **flattened vector** containing information about foods and agents.

Vector size:

max_num_foods * 3 + num_agents * 3

For this environment:

- max_num_foods = 2
- num_agents = 2

Total size = **12 elements**

Layout:

Food slots (first 6 elements):
[Food0_row, Food0_col, Food0_level,
 Food1_row, Food1_col, Food1_level]

Agent slots (last 6 elements):
[Self_row, Self_col, Self_level,
 Teammate_row, Teammate_col, Teammate_level]

Important details:

- The number of food slots is **always fixed** at `max_num_foods = 2`, even if fewer foods exist.
- Missing or eaten foods use sentinel values:

  row = -1  
  col = -1  
  level = 0

- Foods and agents are separated using a **fixed offset**:

  max_num_foods * 3 = 6

  Do **not** dynamically count visible foods.

- The **ego agent (self)** always appears first in the agent slots.

--------------------------------------------------
Coordinate System
--------------------------------------------------

- Coordinates are **raw integer grid indices** (not normalized).
- (0,0) is the **top-left corner**.
- Row index increases **downward**.
- Column index increases **to the right**.

--------------------------------------------------
Action Space
--------------------------------------------------

Discrete action space:

0 : NONE  
1 : NORTH  
2 : SOUTH  
3 : WEST  
4 : EAST  
5 : LOAD  

--------------------------------------------------
Policy Output Requirement
--------------------------------------------------

The policy must return **a single scalar action ID (integer)**.

Correct:
    return torch.tensor(1)

Incorrect:
    return torch.tensor([1])
"""
REWARD_API_REFERENCE = """Reward function API (authoritative):
- Signature: `compute_reward(self, agent)`.
- In LBF training, `self` is the underlying `ForagingEnv` instance (not the VMAS scenario).
- `agent` is one player object from `self.players` corresponding to the currently evaluated agent.
- Accessible state:
  - `self.players`: list of all players.
  - `self.field`: 2D numpy array grid. Cells with value > 0 are foods, and the value is food level.
  - For each player `p` in `self.players`: `p.position`, `p.level`, `p.reward`, `p.score`.
  - `p.reward`: the per-step reward earned this timestep (reset to 0 at the start of each step). Use this to detect whether the agent just collected food.
  - `p.score`: the cumulative reward accumulated over the entire episode (sum of all past `p.reward` values). Use this to measure total episode progress.
- Current agent is `agent` (same attributes as above).
- Important limitations:
  - Use `self.field`, `self.players`, and player attributes to access to states.
  - Do not assume access to history; history-dependent logic is possible only if you explicitly persist state yourself.
- Return value:
  - Return a scalar reward for this single agent at the current step (Python `float` recommended; scalar torch is also accepted)."""

REWARD_EXAMPLE = """
Behavior: move to the nearest food, load it if adjacent, and get a reward boost for successful collection.
```python
import torch

def compute_reward(self, agent):
    reward = 0.0

    # Current agent position for this step.
    row, col = agent.position

    # Find the nearest remaining food from the current grid state.
    nearest_dist = float("inf")
    rows, cols = self.field.shape
    for r in range(rows):
        for c in range(cols):
            if self.field[r, c] > 0:
                dist = abs(r - row) + abs(c - col)
                if dist < nearest_dist:
                    nearest_dist = dist

    if nearest_dist < float("inf"):
        # Dense shaping toward the nearest food.
        reward += 1.0 / (nearest_dist + 1.0)

    # Reinforce successful collection from the environment's score signal.
    if agent.reward > 0:
        reward += 5.0

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
    The action is a single scalar integer (not a list like [1]).
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


def _infer_num_agents(env: gym.Env) -> int:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(base_env, "n_agents"):
        return int(base_env.n_agents)
    if hasattr(base_env, "players"):
        return len(base_env.players)
    if isinstance(env.observation_space, gym.spaces.Tuple):
        return len(env.observation_space.spaces)
    raise ValueError(f"Could not infer number of agents from env type {type(env).__name__}.")


def _build_lbf_group_map(
    task_config: dict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build LBF group map separating scripted teammate from learning agents."""
    n_agents = task_config.get("players")
    if n_agents is None:
        raise ValueError("Could not infer number of agents from LBF task config (expected 'players' key).")
    if n_agents < 2:
        raise ValueError(f"Need at least 2 agents to form ego/teammate split; got {n_agents}.")
    agent_names = [f"agent_{i}" for i in range(n_agents)]
    agents_group = [agent_names[0]]
    teammate_group = agent_names[1:]
    group_map = {"ego": agents_group, "teammate": teammate_group}
    trainable_group = {"ego": agents_group}
    return group_map, trainable_group


def _to_rollout_env(env, device: str = "cpu") -> EnvBase:
    if isinstance(env, TransformedEnv):
        return env
    if isinstance(env, EnvBase):
        return env
    if isinstance(env, gym.Env):
        num_agents = _infer_num_agents(env)
        return LbfGymWrapper(env=env, num_agents=num_agents, device=device)
    raise TypeError(f"Unsupported env type for rollout_lbf_env: {type(env).__name__}")


def _resolve_state_env(env):
    current = env
    visited = set()
    while True:
        if isinstance(current, gym.Env):
            return current
        next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "base_env", None)
        if next_env is None or id(next_env) in visited:
            return env
        visited.add(id(next_env))
        current = next_env



def _collect_step_state(td) -> dict:
    """Extract canonical rollout teams:
    {"ego_team": {...}, "opponent_team": {...}}.
    """
    nested_keys = set(td.keys(include_nested=True))
    groups = [
        key
        for key in td.keys()
        if isinstance(key, str) and (key, "observation") in nested_keys
    ]

    result = {"ego_team": {}, "opponent_team": {}}
    ego_assigned = False
    teammate_idx = 0
    opponent_idx = 0

    for group in groups:
        obs = td[group, "observation"]

        try:
            act = td[group, "action"]
        except KeyError:
            act = None

        try:
            rew = td["next", group, "reward"]
        except KeyError:
            try:
                rew = td[group, "reward"]
            except KeyError:
                rew = None

        obs_2d = obs.reshape(-1, obs.shape[-1])
        num_agents = obs_2d.shape[0]
        act_2d = act.reshape(num_agents, -1) if act is not None else None
        rew_2d = rew.reshape(num_agents, -1) if rew is not None else None

        team = "ego" if group in ("ego", "teammate") else "opponent"
        for i in range(num_agents):
            action_val = None
            if act_2d is not None:
                action_item = act_2d[i].cpu().tolist()
                action_val = action_item[0] if len(action_item) == 1 else action_item

            reward_val = None
            if rew_2d is not None:
                reward_item = rew_2d[i].cpu().tolist()
                reward_val = reward_item[0] if len(reward_item) == 1 else reward_item

            payload = {
                "state": obs_2d[i].cpu().tolist(),
                "action": action_val,
                "reward": reward_val,
            }

            if team == "ego":
                if not ego_assigned:
                    canonical_name = "ego_agent"
                    ego_assigned = True
                else:
                    canonical_name = f"teammate_{teammate_idx}"
                    teammate_idx += 1
                result["ego_team"][canonical_name] = payload
            else:
                canonical_name = f"opponent_{opponent_idx}"
                opponent_idx += 1
                result["opponent_team"][canonical_name] = payload

    return result


def _extract_step_group_rewards(
    rollout_td: TensorDictBase,
    group: str,
    step_idx: int,
    num_steps: int,
) -> list[float]:
    key = ("next", group, "reward")
    if key not in rollout_td.keys(include_nested=True):
        return []

    rew_t = rollout_td[key]
    if rew_t.ndim == 0:
        return [float(rew_t.item())]

    if rew_t.ndim >= 2 and int(rew_t.shape[0]) == int(num_steps):
        rew_step = rew_t[step_idx]
    elif rew_t.ndim >= 2 and int(rew_t.shape[1]) == int(num_steps):
        rew_step = rew_t[:, step_idx]
    elif rew_t.shape[0] > step_idx:
        rew_step = rew_t[step_idx]
    else:
        rew_step = rew_t

    return [float(x) for x in rew_step.reshape(-1).detach().cpu().tolist()]


def _attach_rewards_from_rollout(states: list[dict[str, dict]], rollout_td: TensorDictBase) -> None:
    if not states:
        return

    nested_keys = set(rollout_td.keys(include_nested=True))
    top_groups = [k for k in rollout_td.keys() if isinstance(k, str)]
    reward_groups = [g for g in top_groups if ("next", g, "reward") in nested_keys]
    num_steps = len(states)

    for step_idx, step in enumerate(states):
        ego_names = []
        if "ego_agent" in step.get("ego_team", {}):
            ego_names.append("ego_agent")
        ego_names.extend(
            sorted(
                [k for k in step.get("ego_team", {}) if k.startswith("teammate_")],
                key=lambda x: int(x.split("_")[1]),
            )
        )
        opp_names = sorted(
            [k for k in step.get("opponent_team", {}) if k.startswith("opponent_")],
            key=lambda x: int(x.split("_")[1]),
        )

        ego_vals: list[float] = []
        opp_vals: list[float] = []
        for g in reward_groups:
            vals = _extract_step_group_rewards(rollout_td=rollout_td, group=g, step_idx=step_idx, num_steps=num_steps)
            team = "ego" if g in ("ego", "teammate") else "opponent"
            if team == "ego":
                ego_vals.extend(vals)
            else:
                opp_vals.extend(vals)

        for name, val in zip(ego_names, ego_vals):
            step["ego_team"][name]["reward"] = float(val)
        for name, val in zip(opp_names, opp_vals):
            step["opponent_team"][name]["reward"] = float(val)


def rollout_lbf_env(
    policy,
    env,
    max_steps: int = 50,
    seed: int | None = None,
    **kwargs,
):
    """Rollout a single policy module."""
    rollout_env = _to_rollout_env(env)
    policy.eval()

    def rollout_policy(tensordict):
        out = policy(tensordict)
        if not isinstance(out, TensorDictBase):
            raise TypeError(
                f"Policy output must be TensorDictBase, got {type(out).__name__}."
            )
        return out

    apply_rollout_seed(rollout_env, seed=seed)

    states = []
    images = []

    state_env = _resolve_state_env(rollout_env)
    if hasattr(state_env, "render_mode"):
        try:
            state_env.render_mode = "rgb_array"
        except AttributeError:
            if hasattr(state_env, "unwrapped") and hasattr(state_env.unwrapped, "render_mode"):
                state_env.unwrapped.render_mode = "rgb_array"

    def render_callback(env, td):
        try:
            frame = env.render(mode="rgb_array")
        except TypeError:
            try:
                frame = env.render()
            except Exception:
                frame = None
        except Exception:
            frame = None
        if isinstance(frame, np.ndarray):
            images.append(frame)
        states.append(_collect_step_state(td))

    rollout_sig = inspect.signature(rollout_env.rollout)
    supported_args = set(rollout_sig.parameters.keys())
    rollout_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}
    if "max_steps" in supported_args and "max_steps" not in rollout_kwargs:
        rollout_kwargs["max_steps"] = max_steps

    with torch.no_grad():
        rollout_td = rollout_env.rollout(
            policy=rollout_policy,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_any_done=True,
            **rollout_kwargs,
        )
    _attach_rewards_from_rollout(states=states, rollout_td=rollout_td)

    return states, images, None


def train_lbf_env(
    reward_fn,
    output_dir,
    rl_cfg_dir,
    task_yaml_name="lbf.yaml",
    mappo_yaml_name="mappo.yaml",
    experiment_yaml_name="base_experiment.yaml",
    mlp_yaml_name="mlp.yaml",
    seed=0,
    callbacks=None,
    teammate_policy: list[nn.Module] | None = None,
):
    """Train LBF with optional scripted teammate policies."""
    if hasattr(reward_fn, "__globals__"):
        reward_fn.__globals__["torch"] = torch

    callbacks = list(callbacks or [])

    original_step = LbfGymWrapper._step

    try:
        task = LbfTask.LBF.get_from_yaml(path=str(rl_cfg_dir / task_yaml_name))

        group_map, trainable_group = _build_lbf_group_map(task_config=task.config)
        task.config["group_map"] = group_map

        if teammate_policy is not None:
            from popgen.agents.callback import TeammatePolicyCallback

            callbacks.append(
                TeammatePolicyCallback(
                    teammate_policies=teammate_policy,
                    trainable_group=trainable_group,
                )
            )

        desired_group_map = task.config.get("group_map")
        if desired_group_map:
            name_to_group = {
                aname: (gname, local_i)
                for gname, names in desired_group_map.items()
                for local_i, aname in enumerate(names)
            }

        def custom_step(self, tensordict):
            base_env = self.env.unwrapped
            if desired_group_map:
                action_list = [None] * len(base_env.players)
                for gname, names in desired_group_map.items():
                    g_action = tensordict.get((gname, "action"))
                    for local_i, aname in enumerate(names):
                        global_idx = int(aname.split("_")[1])
                        action_list[global_idx] = int(g_action[local_i].item())
            else:
                action_tensor = tensordict.get(("agents", "action"))
                if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
                    action_idx = action_tensor.argmax(dim=-1)
                else:
                    action_idx = action_tensor
                action_list = action_idx.detach().cpu().numpy().tolist()

            out_td = original_step(self, tensordict)

            new_rewards = []
            for i, player in enumerate(base_env.players):
                player.action = int(action_list[i])
                r = reward_fn(base_env, player)
                if isinstance(r, torch.Tensor):
                    r = r.item()
                new_rewards.append(r)

            if desired_group_map:
                for gname, names in desired_group_map.items():
                    idxs = [int(n.split("_")[1]) for n in names]
                    rt = torch.tensor(
                        [new_rewards[idx] for idx in idxs],
                        device=self.device,
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                    out_td[gname, "reward"] = rt
            else:
                reward_tensor = torch.tensor(
                    new_rewards, device=self.device, dtype=torch.float32
                ).unsqueeze(-1)
                out_td["agents", "reward"] = reward_tensor

            return out_td

        LbfGymWrapper._step = custom_step

        mappo_path = rl_cfg_dir / mappo_yaml_name
        if mappo_path.exists():
            algorithm_config = MappoConfig.get_from_yaml(path=str(mappo_path))
        elif (rl_cfg_dir / "ippo.yaml").exists():
            algorithm_config = IppoConfig.get_from_yaml(path=str(rl_cfg_dir / "ippo.yaml"))
        else:
            algorithm_config = MappoConfig.get_from_yaml()

        experiment_config = ExperimentConfig.get_from_yaml(path=str(rl_cfg_dir / experiment_yaml_name))
        model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / mlp_yaml_name))
        critic_model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / mlp_yaml_name))

        # Disable BenchMARL internal evaluation; B2T evaluates policies via
        # pipeline rollouts/adherence.
        experiment_config.evaluation = False
        experiment_config.evaluation_episodes = 1

        requested_devices = [
            str(getattr(experiment_config, "sampling_device", "")),
            str(getattr(experiment_config, "train_device", "")),
            str(getattr(experiment_config, "buffer_device", "")),
        ]
        if any(dev.startswith("cuda") for dev in requested_devices) and not torch.cuda.is_available():
            logging.warning(
                "CUDA devices were requested in LBF experiment config but CUDA is unavailable. "
                "Falling back to CPU devices for training."
            )
            experiment_config.sampling_device = "cpu"
            experiment_config.train_device = "cpu"
            if str(getattr(experiment_config, "buffer_device", "")).lower() != "disk":
                experiment_config.buffer_device = "cpu"

        experiment_config.save_folder = output_dir
        experiment_config.wandb_extra_kwargs = {"name": Path(output_dir).name, "tags": ["lbf"]}

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks,
        )
        experiment.run()
    finally:
        LbfGymWrapper._step = original_step

    return experiment


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout a single policy module.
    """
    # if env is not defined, load default lbf environment
    if env is None:
        # Foraging-8x8-2p-2f-v3
        env = create_env(**kwargs)

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 50

    return rollout_lbf_env(
        policy=policy,
        env=env,
        **kwargs,
    )


def serialize_state_fn(rollout_data: list[dict]) -> list[dict]:
    for step in rollout_data:
        for team_name in ("ego_team", "opponent_team"):
            for agent_state in step.get(team_name, {}).values():
                raw_state = agent_state.get("state")
                if isinstance(raw_state, list):
                    num_agents = 2
                    num_foods = max(0, (len(raw_state) - num_agents * 3) // 3)
                    foods = []
                    nearest_food_idx = None
                    nearest_food_dist = None
                    nearest_solo_food_idx = None
                    nearest_solo_food_dist = None
                    nearest_joint_food_idx = None
                    nearest_joint_food_dist = None
                    adjacent_food_indices = []

                    self_row = int(raw_state[num_foods * 3])
                    self_col = int(raw_state[num_foods * 3 + 1])
                    self_level = int(raw_state[num_foods * 3 + 2])
                    teammate_row = int(raw_state[num_foods * 3 + 3])
                    teammate_col = int(raw_state[num_foods * 3 + 4])
                    teammate_level = int(raw_state[num_foods * 3 + 5])

                    for food_idx in range(num_foods):
                        row = int(raw_state[3 * food_idx])
                        col = int(raw_state[3 * food_idx + 1])
                        level = int(raw_state[3 * food_idx + 2])
                        present = row >= 0 and col >= 0 and level > 0
                        dist = None
                        adjacent = False
                        teammate_dist = None
                        teammate_adjacent = False
                        solo_feasible = False
                        jointly_feasible = False
                        if present:
                            dist = abs(row - self_row) + abs(col - self_col)
                            adjacent = dist == 1
                            teammate_dist = abs(row - teammate_row) + abs(col - teammate_col)
                            teammate_adjacent = teammate_dist == 1
                            solo_feasible = self_level >= level
                            jointly_feasible = (self_level + teammate_level) >= level
                            if adjacent:
                                adjacent_food_indices.append(food_idx)
                            if nearest_food_dist is None or dist < nearest_food_dist:
                                nearest_food_dist = dist
                                nearest_food_idx = food_idx
                            if solo_feasible and (nearest_solo_food_dist is None or dist < nearest_solo_food_dist):
                                nearest_solo_food_dist = dist
                                nearest_solo_food_idx = food_idx
                            if jointly_feasible and (nearest_joint_food_dist is None or dist < nearest_joint_food_dist):
                                nearest_joint_food_dist = dist
                                nearest_joint_food_idx = food_idx
                        foods.append(
                            {
                                "slot": food_idx,
                                "present": present,
                                "pos": [row, col] if present else None,
                                "level": level,
                                "distance": dist,
                                "adjacent": adjacent,
                                "teammate_distance": teammate_dist,
                                "teammate_adjacent": teammate_adjacent,
                                "solo_feasible": solo_feasible,
                                "jointly_feasible": jointly_feasible,
                                "requires_teammate": present and not solo_feasible and jointly_feasible,
                                "uncollectable_even_together": present and not jointly_feasible,
                                "cooperative_load_ready": present and adjacent and teammate_adjacent and jointly_feasible,
                            }
                        )

                    agent_state["serialized_state"] = {
                        "ego_pos": [self_row, self_col],
                        "ego_level": self_level,
                        "teammate_pos": [teammate_row, teammate_col],
                        "teammate_level": teammate_level,
                        "teammate_distance": abs(self_row - teammate_row) + abs(self_col - teammate_col),
                        "foods": foods,
                        "visible_food_count": sum(1 for food in foods if food["present"]),
                        "nearest_food_slot": nearest_food_idx,
                        "nearest_food_distance": nearest_food_dist,
                        "nearest_solo_food_slot": nearest_solo_food_idx,
                        "nearest_solo_food_distance": nearest_solo_food_dist,
                        "nearest_joint_food_slot": nearest_joint_food_idx,
                        "nearest_joint_food_distance": nearest_joint_food_dist,
                        "adjacent_food_slots": adjacent_food_indices,
                        "load_available": bool(adjacent_food_indices),
                        "solo_load_ready": any(food["adjacent"] and food["solo_feasible"] for food in foods),
                        "cooperative_load_ready": any(food["cooperative_load_ready"] for food in foods),
                    }
    return rollout_data


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
            scripted_policy = ScriptedPolicy(temp_env)
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

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Select action for each observation in the batch.

        Args:
            obs_tensor: Observation tensor with shape ``(batch, obs_dim)``.

        Returns:
            Action tensor, shape ``(batch,)`` with integer action indices.
        """
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(f"LBF ScriptedPolicy expected torch.Tensor observation, got {type(obs_tensor).__name__}.")
        if obs_tensor.dim() != 2:
            raise ValueError(
                f"LBF ScriptedPolicy expects observation tensor with shape (batch, obs_dim), got {tuple(obs_tensor.shape)}."
            )
        batch_size = obs_tensor.shape[0]

        actions = torch.zeros(batch_size, dtype=torch.long, device=obs_tensor.device)

        for b in range(batch_size):
            o = obs_tensor[b]

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

        return actions


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
    task = LbfTask.LBF.get_from_yaml(path=str(RL_CFG_DIR / "lbf.yaml"))
    group_map, _ = _build_lbf_group_map(task_config=task.config)
    task.config["group_map"] = group_map
    return task.get_env_fun(
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
    )()
