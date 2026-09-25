import inspect
import logging
import os
import random
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Enable headless rendering for pyglet (used by lbforaging) before importing lbforaging.
os.environ.setdefault("PYGLET_HEADLESS", "true")

import gymnasium as gym

# Ensure lbforaging is registered
import lbforaging  # noqa: F401
import popgen.envs.lbf.register_envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from benchmarl.environments.lbf.common import LbfGymWrapper, LbfTask
from benchmarl.experiment import Experiment, ExperimentConfig
from pydantic import BaseModel, Field
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, TransformedEnv

from popgen.envs.utils import apply_rollout_seed, apply_train_kwargs, load_algorithm_config, load_model_config, resolve_cfg_dir

_LBF_RENDER_LOCK = threading.Lock()


def _patch_pyglet_vertexdomain_finalizer() -> None:
    """Guard pyglet finalization for partially initialized render batches."""
    try:
        from pyglet.graphics.vertexdomain import VertexDomain
    except Exception:
        return

    if getattr(VertexDomain, "_popgen_safe_del", False):
        return

    original_del = VertexDomain.__del__

    def _safe_del(self):
        if not hasattr(self, "attributes"):
            return
        try:
            original_del(self)
        except Exception:
            pass

    VertexDomain.__del__ = _safe_del
    VertexDomain._popgen_safe_del = True

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
- Every food in the environment should be cooperatively loaded by the agents together, as no single agent can carry any food alone.

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
  - For each player `p` in `self.players`: `p.position`, `p.level`, `p.reward`, `p.score`, `p.action`.
  - Current agent is `agent` (same attributes as above).
- Important limitations:
  - Use `self.field`, `self.players`, and player attributes to access to states.
  - Do not assume access to history; history-dependent logic is possible only if you explicitly persist state yourself.
- Return value:
  - Return a scalar reward for this single agent at the current step (Python `float` recommended; scalar torch is also accepted).
  - Keep the per-step reward within [-10, 10]. Values outside that range, and any NaN or inf, are clamped at runtime.
  - Keep dense per-step terms small, around |r| <= 1, and reserve larger values for rare events (order 1-5). The clamp bounds a single step, not the episode: a dense term of 1 per step already accumulates to hundreds over an episode, and returns that large make advantages explode, which drives the policy to NaN and kills the run. Aim for an episode return in the low tens.
  - Prefer normalized or tanh-shaped dense terms over raw distances or unbounded sums."""

REWARD_EXAMPLE = """
Behavior: move to the nearest food, load it if adjacent, and avoid idling
```python
import torch

def compute_reward(self, agent):
    reward = 0.0

    # Current agent position and chosen discrete action for this step.
    row, col = agent.position
    action = agent.action

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

        # Encourage LOAD when already in a valid loading position.
        if nearest_dist == 1 and action == 5:
            reward += 2.0

    # Reinforce successful collection from the environment's score signal.
    if agent.reward > 0 or agent.score > 0:
        reward += 5.0

    # Small penalty for doing nothing when not adjacent to food.
    if action == 0 and nearest_dist > 1:
        reward -= 0.05

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


def build_group_map(
    task_config: dict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build LBF training group map separating scripted teammate from learning agents."""
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


def _reset_lbf_renderer(env) -> None:
    """Close and clear the lbforaging viewer so later rollouts get a fresh context."""
    state_env = _resolve_state_env(env)
    viewer = getattr(state_env, "viewer", None)
    if viewer is not None:
        with _LBF_RENDER_LOCK:
            try:
                viewer.close()
            except Exception:
                pass
        state_env.viewer = None
    if hasattr(state_env, "_rendering_initialized"):
        state_env._rendering_initialized = False



def _group_to_team(group_name: str, all_groups: list[str]) -> str:
    """Map env-specific group names to canonical team buckets."""
    group = str(group_name).lower()
    lower_groups = {str(g).lower() for g in all_groups}

    if group in {"ego", "teammate", "agents", "agent_blue", "blue"}:
        return "ego"
    if group in {"opponent", "opponents", "agent_red", "red", "enemy", "enemies"}:
        return "opponent"

    if "ego" in lower_groups or "teammate" in lower_groups:
        return "opponent" if group not in {"ego", "teammate"} else "ego"

    if "agent_blue" in lower_groups or "agent_red" in lower_groups:
        return "opponent" if group == "agent_red" else "ego"

    if any(tok in group for tok in ("opponent", "enemy", "red")):
        return "opponent"
    return "ego"


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

        team = _group_to_team(group_name=group, all_groups=groups)
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

    # rollout_td has batch_size (num_envs, T) when returned by env.rollout(),
    # so rew_t has shape (num_envs, T, n_agents, reward_dim).
    # Index using batch_dims to correctly extract the reward at step_idx.
    batch_dims = rollout_td.batch_dims
    T = rollout_td.shape[-1] if batch_dims >= 1 else 0

    if batch_dims == 2:
        if step_idx >= T:
            return []
        rew_step = rew_t[0, step_idx]  # (n_agents, reward_dim)
    elif batch_dims == 1:
        if step_idx >= T:
            return []
        rew_step = rew_t[step_idx]  # (n_agents, reward_dim)
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
            team = _group_to_team(group_name=g, all_groups=reward_groups)
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
    """Rollout a policy in the LBF environment."""
    rollout_env = _to_rollout_env(env)
    policy.eval()
    _patch_pyglet_vertexdomain_finalizer()

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
        with _LBF_RENDER_LOCK:
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

    try:
        with torch.no_grad():
            rollout_td = rollout_env.rollout(
                policy=rollout_policy,
                callback=render_callback,
                auto_cast_to_device=True,
                break_when_any_done=True,
                **rollout_kwargs,
            )
        # Recover any steps the callback missed (typically the terminal step).
        batch_dims = rollout_td.batch_dims
        T = rollout_td.shape[-1] if batch_dims >= 1 else 0
        for t in range(len(states), T):
            step_td = rollout_td[:, t] if batch_dims == 2 else rollout_td[t : t + 1]
            states.append(_collect_step_state(step_td))

        _attach_rewards_from_rollout(states=states, rollout_td=rollout_td)
    finally:
        _reset_lbf_renderer(rollout_env)

    return states, images, None


def train_lbf_env(
    reward_fn,
    output_dir,
    rl_cfg_dir,
    run_name: str | None = None,
    task_yaml_name="lbf.yaml",
    experiment_yaml_name="base_experiment.yaml",
    seed=0,
    callbacks=None,
    teammate_policy: list[nn.Module] | None = None,
    **kwargs,
):
    """Train LBF with optional scripted teammate policies."""
    if "centralized_critic" in kwargs:
        raise ValueError(
            "centralized_critic is not supported. Only IPPO-style (non-centralized) training is available."
        )
    prioritized: bool = kwargs.pop("prioritized", True)

    if hasattr(reward_fn, "__globals__"):
        reward_fn.__globals__["torch"] = torch

    callbacks = list(callbacks or [])
    base_cfg_dir = Path(rl_cfg_dir)
    rl_cfg_dir = resolve_cfg_dir(base_cfg_dir, teammate_policy)

    original_step = LbfGymWrapper._step

    try:
        task = LbfTask.LBF.get_from_yaml(path=str(base_cfg_dir / task_yaml_name))

        if teammate_policy is not None:
            group_map, trainable_group = build_group_map(task_config=task.config)
            task.config["group_map"] = group_map

            is_population = isinstance(teammate_policy[0], list)
            from popgen.agents.callbacks import select_teammate_callback_cls

            callback_cls = select_teammate_callback_cls(
                is_population=is_population,
                prioritized=prioritized,
            )
            callbacks.append(callback_cls(teammate_policies=teammate_policy, trainable_group=trainable_group))

        if reward_fn is not None:
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
                    if desired_group_map:
                        gname, local_i = name_to_group[f"agent_{i}"]
                        player.obs = out_td[gname, "observation"][local_i].cpu().numpy()
                    else:
                        player.obs = out_td["agents", "observation"][i].cpu().numpy()
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

        algorithm_config = load_algorithm_config(rl_cfg_dir)
        experiment_config = ExperimentConfig.get_from_yaml(path=str(rl_cfg_dir / experiment_yaml_name))
        model_config = load_model_config(rl_cfg_dir)
        critic_model_config = load_model_config(rl_cfg_dir)

        experiment_config.render = True

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

        apply_train_kwargs(kwargs, experiment_config, algorithm_config, model_config, critic_model_config)

        experiment_config.save_folder = output_dir
        display_run_name = run_name.strip() if isinstance(run_name, str) and run_name.strip() else Path(output_dir).name
        experiment_config.wandb_extra_kwargs = {"name": display_run_name, "tags": ["lbf"]}

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


def train_fn(
    reward_fn: Callable | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    **kwargs,
):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward (float or tensor).
        output_dir: Directory to save results.
        run_name: Name for the training run.
    """
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be provided and non-empty.")

    teammate_policy = kwargs.pop("teammate_policy", None)
    config_overrides = kwargs.pop("config_overrides", None)

    if teammate_policy is None:
        temp_env = create_env(num_envs=1, seed=0, device="cpu")
        try:
            teammate_policy = [ClosestFoodPolicy(temp_env)]
        finally:
            temp_env.close()

    return train_lbf_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        run_name=run_name.strip(),
        rl_cfg_dir=RL_CFG_DIR,
        teammate_policy=teammate_policy,
        config_overrides=config_overrides,
        **kwargs,
    )

# ---------------------------------------------------------------------------- #
#                              Scripted Policy                                 #
# ---------------------------------------------------------------------------- #


class _LbfScriptedPolicyBase(torch.nn.Module):
    """Base class for scripted teammate policies in LBF."""

    # Action constants
    NONE = 0
    NORTH = 1  # row - 1
    SOUTH = 2  # row + 1
    WEST = 3   # col - 1
    EAST = 4   # col + 1
    LOAD = 5
    MOVEMENT_RANDOM_EPSILON = 0.2

    def __init__(self, env):
        super().__init__()
        self.num_agents, self.num_foods = _extract_env_metadata(env)

    @staticmethod
    def _manhattan(row_a: float, col_a: float, row_b: float, col_b: float) -> float:
        return abs(row_a - row_b) + abs(col_a - col_b)

    def _parse_observation_row(
        self,
        row_obs: torch.Tensor,
    ) -> tuple[float, float, float, list[tuple[float, float, float]], list[tuple[float, float, float, int]]]:
        foods: list[tuple[float, float, float, int]] = []
        for food_idx in range(self.num_foods):
            food_row = float(row_obs[3 * food_idx].item())
            food_col = float(row_obs[3 * food_idx + 1].item())
            food_level = float(row_obs[3 * food_idx + 2].item())
            if food_row < 0 or food_col < 0 or food_level <= 0:
                continue
            foods.append((food_row, food_col, food_level, food_idx))

        agent_offset = self.num_foods * 3
        self_row = float(row_obs[agent_offset].item())
        self_col = float(row_obs[agent_offset + 1].item())
        self_level = float(row_obs[agent_offset + 2].item())

        teammates: list[tuple[float, float, float]] = []
        for agent_idx in range(1, self.num_agents):
            offset = agent_offset + 3 * agent_idx
            if offset + 2 >= row_obs.shape[0]:
                break
            tm_row = float(row_obs[offset].item())
            tm_col = float(row_obs[offset + 1].item())
            tm_level = float(row_obs[offset + 2].item())
            if tm_row < 0 or tm_col < 0 or tm_level <= 0:
                continue
            teammates.append((tm_row, tm_col, tm_level))

        return self_row, self_col, self_level, teammates, foods

    def _move_towards(self, self_row: float, self_col: float, target_row: float, target_col: float) -> int:
        row_diff = target_row - self_row
        col_diff = target_col - self_col
        if abs(row_diff) < 1e-9 and abs(col_diff) < 1e-9:
            return self.NONE

        vertical_action = None
        if abs(row_diff) >= 1e-9:
            vertical_action = self.NORTH if row_diff < 0 else self.SOUTH

        horizontal_action = None
        if abs(col_diff) >= 1e-9:
            horizontal_action = self.WEST if col_diff < 0 else self.EAST

        if vertical_action is None:
            greedy_action = horizontal_action if horizontal_action is not None else self.NONE
        elif horizontal_action is None:
            greedy_action = vertical_action
        elif abs(row_diff) > abs(col_diff):
            greedy_action = vertical_action
        elif abs(col_diff) > abs(row_diff):
            greedy_action = horizontal_action
        else:
            greedy_action = vertical_action

        epsilon = min(max(float(self.MOVEMENT_RANDOM_EPSILON), 0.0), 1.0)
        if random.random() < epsilon:
            return random.choice((self.NORTH, self.SOUTH, self.WEST, self.EAST))
        return greedy_action

    def _choose_target_food(
        self,
        self_row: float,
        self_col: float,
        self_level: float,
        teammates: list[tuple[float, float, float]],
        foods: list[tuple[float, float, float, int]],
    ) -> tuple[float, float, float, int] | None:
        raise NotImplementedError

    def _adjacent_action(
        self,
        self_row: float,
        self_col: float,
        target_row: float,
        target_col: float,
        teammates: list[tuple[float, float, float]],
    ) -> int:
        del self_row, self_col, target_row, target_col, teammates
        return self.LOAD

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"LBF ScriptedPolicy expected torch.Tensor observation, got {type(obs).__name__}.")
        if obs.dim() != 2:
            raise ValueError(f"LBF ScriptedPolicy expects observation tensor with shape (batch, obs_dim), got {tuple(obs.shape)}.")

        batch_size = obs.shape[0]
        actions = torch.full((batch_size,), self.NONE, dtype=torch.long, device=obs.device)

        for batch_idx in range(batch_size):
            self_row, self_col, self_level, teammates, foods = self._parse_observation_row(obs[batch_idx])
            if self_row < 0 or self_col < 0 or not foods:
                actions[batch_idx] = self.NONE
                continue

            target = self._choose_target_food(self_row, self_col, self_level, teammates, foods)
            if target is None:
                actions[batch_idx] = self.NONE
                continue

            target_row, target_col, _target_level, _target_idx = target
            dist_to_target = self._manhattan(self_row, self_col, target_row, target_col)

            if dist_to_target <= 1:
                actions[batch_idx] = self._adjacent_action(
                    self_row=self_row,
                    self_col=self_col,
                    target_row=target_row,
                    target_col=target_col,
                    teammates=teammates,
                )
                continue

            actions[batch_idx] = self._move_towards(self_row, self_col, target_row, target_col)

        return actions


class ClosestFoodPolicy(_LbfScriptedPolicyBase):
    """Greedy partner: always move toward the nearest visible food."""

    def _choose_target_food(
        self,
        self_row: float,
        self_col: float,
        self_level: float,
        teammates: list[tuple[float, float, float]],
        foods: list[tuple[float, float, float, int]],
    ) -> tuple[float, float, float, int] | None:
        del self_level, teammates
        return min(
            foods,
            key=lambda food: (self._manhattan(self_row, self_col, food[0], food[1]), food[3]),
        )


class TeamCenterFoodPolicy(_LbfScriptedPolicyBase):
    """Center-seeking partner: pick food closest to team center position."""

    def _choose_target_food(
        self,
        self_row: float,
        self_col: float,
        self_level: float,
        teammates: list[tuple[float, float, float]],
        foods: list[tuple[float, float, float, int]],
    ) -> tuple[float, float, float, int] | None:
        del self_level
        team_rows = [self_row] + [teammate[0] for teammate in teammates]
        team_cols = [self_col] + [teammate[1] for teammate in teammates]
        center_row = sum(team_rows) / len(team_rows)
        center_col = sum(team_cols) / len(team_cols)
        return min(
            foods,
            key=lambda food: (self._manhattan(center_row, center_col, food[0], food[1]), food[3]),
        )


class TeammateBiasedFoodPolicy(_LbfScriptedPolicyBase):
    """Supportive partner: prioritize food that minimizes teammate travel."""

    def _choose_target_food(
        self,
        self_row: float,
        self_col: float,
        self_level: float,
        teammates: list[tuple[float, float, float]],
        foods: list[tuple[float, float, float, int]],
    ) -> tuple[float, float, float, int] | None:
        del self_level
        if not teammates:
            return min(
                foods,
                key=lambda food: (self._manhattan(self_row, self_col, food[0], food[1]), food[3]),
            )

        return min(
            foods,
            key=lambda food: (
                min(self._manhattan(teammate[0], teammate[1], food[0], food[1]) for teammate in teammates),
                self._manhattan(self_row, self_col, food[0], food[1]),
                food[3],
            ),
        )


class SynchronizedLoadingFoodPolicy(_LbfScriptedPolicyBase):
    """Coordination-focused partner: wait at food until teammate is also adjacent."""

    def _choose_target_food(
        self,
        self_row: float,
        self_col: float,
        self_level: float,
        teammates: list[tuple[float, float, float]],
        foods: list[tuple[float, float, float, int]],
    ) -> tuple[float, float, float, int] | None:
        del self_level
        return min(
            foods,
            key=lambda food: (
                self._manhattan(self_row, self_col, food[0], food[1])
                + sum(self._manhattan(teammate[0], teammate[1], food[0], food[1]) for teammate in teammates),
                food[3],
            ),
        )

    def _adjacent_action(
        self,
        self_row: float,
        self_col: float,
        target_row: float,
        target_col: float,
        teammates: list[tuple[float, float, float]],
    ) -> int:
        del self_row, self_col
        all_teammates_adjacent = all(
            self._manhattan(teammate[0], teammate[1], target_row, target_col) <= 1
            for teammate in teammates
        )
        if not teammates or all_teammates_adjacent:
            return self.LOAD
        return self.NONE


# Backward compatibility for existing codepaths expecting env_module.ScriptedPolicy.
ScriptedPolicy = ClosestFoodPolicy


EVALUATION_BEHAVIOR_SET: list[dict[str, Any]] = [
    {
        "name": "closest_food_greedy",
        "description": "Greedy nearest-food teammate baseline.",
        "policy_cls": ClosestFoodPolicy,
    },
    {
        "name": "team_center_food",
        "description": "Targets foods near team center to reduce split behavior.",
        "policy_cls": TeamCenterFoodPolicy,
    },
    {
        "name": "teammate_biased_food",
        "description": "Prioritizes foods that are easier for teammate to reach.",
        "policy_cls": TeammateBiasedFoodPolicy,
    },
    {
        "name": "synchronized_loading_food",
        "description": "Waits at target food until teammate is adjacent, then loads.",
        "policy_cls": SynchronizedLoadingFoodPolicy,
    },
]

def get_evaluation_scripted_policies() -> list[Callable[[Any], torch.nn.Module]]:
    """Return scripted policy factories for evaluation behavior sweeps."""
    return [entry["policy_cls"] for entry in EVALUATION_BEHAVIOR_SET]


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
    group_map, _ = build_group_map(task.config)
    task.config["group_map"] = group_map
    return task.get_env_fun(
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
    )()
