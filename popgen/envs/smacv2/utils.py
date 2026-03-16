"""SMACv2 `terran_2_vs_4` observation and action execution reference.

This module is configured for 2 allied marines vs 4 enemy marines.

Observation space (per-agent, 50 dims in this config):
- move_feats: 4 dims (can move north/south/east/west)
- enemy_feats: 4 enemies x 8 dims each = 32 dims
  - per enemy: shootable, distance/sight_range, relative_x/sight_range,
    relative_y/sight_range, health, and 3 unit-type bits
- ally_feats: 1 ally x 8 dims = 8 dims
- own_feats: 6 dims (own health, normalized x/y, own unit-type bits)

Out-of-sight enemy handling:
- This task uses `conic_fov=False` and marine sight range 9, so visibility is
  distance-based (`distance < sight_range`).
- With `prob_obs_enemy=1.0`, if one ally observes an enemy, that enemy is
  shared to the other ally's local observation.
- Enemies that are out of sight (or dead) appear as zeros in enemy features
  for that slot (`shootable=0`, `distance=0`, `relative_x/y=0`, `health=0`,
  unit-type bits all zero).

Unit-type bits (`unit_type_bits=3` on `10gen_terran`):
- one-hot unit identity channels from `get_unit_type_id` in SMACv2 source.
- terran_gen mapping is: marine -> bit_0, marauder -> bit_1, medivac -> bit_2.
- in `terran_2_vs_4`, all units are marines, so visible unit-type bits are
  typically `[1, 0, 0]`.
- when an enemy is not visible/dead, its whole enemy feature row is zeroed, so
  its type bits are also all zeros.

Action availability and execution (`action_mask: False` in this config):
- 0: no-op (valid for dead agents)
- 1: stop
- 2..5: movement (north, south, east, west)
- 6+: attack enemy_k (offset by `n_actions_no_attack`)
- `action_mask` tensor is still present in tensordict, but with this setting attack slots
  are enabled for alive enemies even when they are out of shoot range.
- If the policy emits an action that cannot be executed (for example, attack out of range),
  SMACv2 does not execute that command for that step.

Where to verify these details:
- `popgen/envs/smacv2/rl_cfg/terran_2_vs_4.yaml`
  - scenario config (`n_units`, `n_enemies`, observation flags, `action_mask: False`)
- `deps/smacv2/smacv2/env/starcraft2/starcraft2.py`
  - `get_obs_agent`: exact observation feature construction/order
  - `get_obs_enemy_feats_size`, `get_obs_move_feats_size`: feature dimensions
  - `n_actions_no_attack`: where attack action indices start
  - `get_avail_agent_actions`: available-action construction
  - `get_agent_action`: execution-time checks (commands may be skipped)
- `scripts/smacv2_test.py`
  - prints observation feature names/values during rollout for inspection.
"""

import inspect
import logging as py_logging
import os
import re
import subprocess
from copy import deepcopy
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch
import torch.nn as nn
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments import Smacv2Task
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs.libs.smacv2 import SMACv2Env
from torchrl.envs.utils import ACTION_MASK_ERROR

from popgen.agents.utils import (
    extract_observation_and_action_key,
    find_observation_group,
    flatten_obs,
    format_action_output,
)
from popgen.agents.scripted_partner import MultiPolicyWrapper
from popgen.envs.utils import resolve_rollout_seed


_ACTIVE_SMACV2_REWARD_FN: Callable | None = None

def compose_rollout_policy(policy_or_policies):
    """Normalize rollout input to a single policy module."""
    if isinstance(policy_or_policies, (list, tuple)):
        if len(policy_or_policies) == 1:
            policy = policy_or_policies[0]
        elif len(policy_or_policies) == 2:
            policy = MultiPolicyWrapper(
                policies=[policy_or_policies[0], policy_or_policies[1]],
            )
        else:
            raise ValueError(
                f"Expected 1 or 2 policies for rollout, got {len(policy_or_policies)}."
            )
    else:
        policy = policy_or_policies

    if not hasattr(policy, "eval"):
        raise TypeError(
            f"Policy must be nn.Module-like with eval(), got {type(policy).__name__}."
        )

    return policy


def _normalize_nested_key(key) -> tuple[str, ...]:
    if isinstance(key, tuple):
        return tuple(str(part) for part in key)
    return (str(key),)


def _iter_reward_leaf_paths(step_td) -> list[tuple[str, ...]]:
    try:
        keys = step_td.keys(include_nested=True, leaves_only=True)
    except TypeError:
        keys = step_td.keys(include_nested=True)

    reward_paths: list[tuple[str, ...]] = []
    for key in keys:
        path = _normalize_nested_key(key)
        if path and path[-1] == "reward":
            reward_paths.append(path)
    return reward_paths


def _select_reward_paths_by_group(reward_paths: list[tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    selected: dict[str, tuple[int, tuple[str, ...]]] = {}
    for path in reward_paths:
        if path[0] == "next" and len(path) >= 3:
            group = path[1]
            priority = 0
        elif len(path) >= 2:
            group = path[0]
            priority = 1
        else:
            group = "root"
            priority = 2

        previous = selected.get(group)
        if previous is None or priority < previous[0]:
            selected[group] = (priority, path)

    return {group: entry[1] for group, entry in selected.items()}


def _reward_value_mean(value) -> float | None:
    if value is None:
        return None

    try:
        if hasattr(value, "float") and callable(value.float):
            mean_value = value.float().mean()
        elif hasattr(value, "mean") and callable(value.mean):
            mean_value = value.mean()
        else:
            return float(value)

        if hasattr(mean_value, "item") and callable(mean_value.item):
            return float(mean_value.item())
        return float(mean_value)
    except Exception:
        return None


def extract_reward_rollout_info(rollout_td) -> dict:
    """Aggregate rollout rewards by group from a rollout TensorDict-like object."""
    reward_totals_by_group: dict[str, float] = {}
    reward_steps = 0

    rollout_batch_size = getattr(rollout_td, "batch_size", ())

    if len(rollout_batch_size) >= 2:
        rollout_td = rollout_td[0]
        rollout_batch_size = rollout_td.batch_size

    n_steps = int(rollout_batch_size[0]) if rollout_batch_size else 0

    for step_idx in range(n_steps):
        step_td = rollout_td[step_idx]
        reward_paths = _iter_reward_leaf_paths(step_td)
        selected_paths = _select_reward_paths_by_group(reward_paths)
        if not selected_paths:
            continue

        step_has_reward = False
        for group, path in selected_paths.items():
            try:
                reward_value = _reward_value_mean(step_td[path])
            except Exception:
                reward_value = None

            if reward_value is None:
                continue

            reward_totals_by_group[group] = reward_totals_by_group.get(group, 0.0) + reward_value
            step_has_reward = True

        if step_has_reward:
            reward_steps += 1

    reward_total_all_groups = float(sum(reward_totals_by_group.values()))
    return {
        "reward_totals_by_group": reward_totals_by_group,
        "reward_total_all_groups": reward_total_all_groups,
        "reward_steps": reward_steps,
    }


# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

ENVIRONMENT_DESCRIPTION = """The environment is called "SMACv2" (StarCraft Multi-Agent Challenge v2).
It is a cooperative MARL benchmark built on StarCraft II.

This configuration uses the "terran_2_vs_4" scenario: 2 allied Terran units must coordinate to defeat 4 enemy units.
All allied and enemy units are Marines.

The team wins by eliminating all enemy units. Episodes end when all enemies are defeated,
all allies die, or the episode step limit is reached.

> **Observation Space**:
For this `terran_2_vs_4` setup, each agent receives a **50-dimensional** local observation vector.
It is a concatenation of:
- `move_feats` (4 dims): availability of moving north/south/east/west.
- `enemy_feats` (4 enemies x 8 dims = 32):
  - per enemy: `shootable`, `distance/sight_range`, `relative_x/sight_range`, `relative_y/sight_range`,
    `health`, and 3 unit-type bits.
- `ally_feats` (1 ally x 8 dims = 8):
  - for the other allied unit: `visible`, `distance/sight_range`, `relative_x/sight_range`,
    `relative_y/sight_range`, `health`, and 3 unit-type bits.
- `own_feats` (6 dims):
  - own `health`, normalized `pos_x` and `pos_y`, and 3 own unit-type bits.

Important details for this config:
- Range settings: `use_unit_ranges=True` and `min_attack_range=2`.
  In this scenario all controllable units are marines, so:
  - shoot range = `max(5, min_attack_range)` = **5**
  - sight range = **9**
- Reward-function object model:
  - `env.agents` and `env.enemies` are dictionaries keyed by integer unit ids.
  - Access the controlled marine as `env.agents[agent_id]`.
  - Iterate enemies with `for enemy_id, enemy in env.enemies.items():`.
  - Unit coordinates live at `unit.pos.x` and `unit.pos.y`.
    Do not assume `unit.pos_x` / `unit.pos_y` on the raw SMAC units.
  - Previous-step snapshots are stored in `env.previous_ally_units` and
    `env.previous_enemy_units`, keyed the same way as `env.agents` / `env.enemies`.
- Exact normalization used in observations:
  - enemy distance feature = `euclidean_distance(self, enemy) / sight_range`
  - enemy relative coordinates = `(enemy_x - self_x) / sight_range`,
    `(enemy_y - self_y) / sight_range`
  - ally distance/relative coordinates use the same sight-range normalization
  - own health feature = `health / health_max`
  - own position features = `x / map_x` and `y / map_y`
    (this task uses a 32x32 start-position distribution)
- Out-of-sight enemy behavior:
  - enemy visibility is based on `distance < sight_range`
  - with `prob_obs_enemy=1.0`, observations of visible enemies are shared across allies
  - out-of-sight/dead enemy slots are zero-filled in `enemy_feats`
- `action_mask=False`, so attack actions can appear available for alive enemies even when
  they are out of range or out of sight; those non-executable commands are skipped by the env.
- Unit-type bits are a one-hot 3-bit encoding (`marine`, `marauder`, `medivac`) from SMACv2's
  `get_unit_type_id` logic. In this specific scenario all units are marines, so the active bit is
  usually `bit_0` for visible units; unseen/dead enemy rows remain zero.

> **Action Space**:
The action space is discrete with dynamic size depending on the number of visible enemies:
- Action 0: no-op (dead agents only)
- Actions 1-5: movement (stop, north, south, east, west)
- Actions 6+: attack enemy k (for each currently visible enemy)

Attack actions can appear available for alive enemies even if they are out of shoot range.
If the policy emits an action that cannot be executed (e.g., out-of-range attack), the command is not executed.
"""

REWARD_API_REFERENCE = """Reward function API (authoritative):
- Signature: `compute_reward(env, agent_id)`.
- In SMACv2 training, `env` is a reward-compatible proxy around the underlying
  SMAC2 environment instance. All normal env methods still work.
- `agent_id` is an integer index (0 to n_agents-1) of the agent being evaluated.
- Accessible state via `env`:
  - `env.get_obs_agent(agent_id)`: numpy array of agent's observation.
  - `env.get_obs()`: list of observations for all agents.
  - `env.get_state()`: global state array.
  - `env.get_avail_agent_actions(agent_id)`: list of available action booleans.
  - `env.agents`: `dict[int, unit]` for allied units.
    Use `env.agents[agent_id]` for the controlled marine.
  - `env.enemies`: `dict[int, unit]` for enemy units.
    Iterate with `env.enemies.items()` or `env.enemies.values()`.
  - unit fields commonly useful in rewards:
    - `unit.health`, `unit.health_max`
    - `unit.shield`, `unit.shield_max`
    - `unit.pos.x`, `unit.pos.y`
    - `unit.tag`, `unit.unit_type`
  - previous-step snapshots:
    - `env.previous_ally_units[agent_id]`
    - `env.previous_enemy_units[enemy_id]`
  - compatibility fields added by the proxy:
    - `unit.previous_health`
    - `unit.is_alive`
    - `unit.pos_x`, `unit.pos_y`
- Important restrictions:
  - Do not assume `env.agents` or `env.enemies` are lists.
  - Do not invent undocumented attributes such as `unit.action_recent`.
- Return value:
  - Return a scalar reward for this single agent at the current step (Python `float` recommended; scalar torch is also accepted)."""

REWARD_EXAMPLE = """
Behavior: reward agent for surviving, keeping kite distance, and dealing damage
```python
def compute_reward(env, agent_id):
    agent = env.agents[agent_id]
    if not agent.is_alive:
        return -1.0

    reward = 0.05

    # Reward staying near shooting distance from the closest living enemy.
    nearest_enemy_dist = None
    for enemy_id, enemy in env.enemies.items():
        if enemy.health <= 0:
            continue
        dx = enemy.pos.x - agent.pos.x
        dy = enemy.pos.y - agent.pos.y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if nearest_enemy_dist is None or dist < nearest_enemy_dist:
            nearest_enemy_dist = dist

        # Reward enemy health decreases using the previous-step snapshot.
        prev_enemy = env.previous_enemy_units.get(enemy_id)
        if prev_enemy is not None and enemy.health < prev_enemy.health:
            reward += 0.02 * (prev_enemy.health - enemy.health)

    if nearest_enemy_dist is not None:
        if 3.0 <= nearest_enemy_dist <= 5.0:
            reward += 0.05
        elif nearest_enemy_dist < 3.0:
            reward -= 0.1

    return float(reward)
```
"""

# ---------------------------------------------------------------------------- #
#                                 Action Type                                  #
# ---------------------------------------------------------------------------- #


class ActionType(BaseModel):
    """
    Action type for the SMACv2 environment.
    - action_id: discrete action index (scalar), not one-hot.

    Action meanings depend on visible enemies each step, but the general layout is:
    0: no-op (only valid for dead agents)
    1: stop
    2: move north
    3: move south
    4: move east
    5: move west
    6+: attack enemy k (for visible enemy k)

    Example: to move north, return ActionType(action_id=2).
    """

    action_id: int = Field(default=0, description="Discrete action index for SMACv2.")

    @classmethod
    def default_action(cls):
        """Default action (no-op)."""
        return cls(action_id=0)

    def to_tensor(self):
        """Convert action to a scalar torch tensor."""
        return torch.tensor(self.action_id, dtype=torch.int64)


# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #


class _FixedGroupMap:
    """Picklable callable that returns a fixed group_map regardless of env."""

    def __init__(self, group_map: dict):
        self._group_map = group_map

    def __call__(self, env):
        return self._group_map


def _previous_unit_health(previous_units: dict | None, unit_id: int, current_unit) -> float:
    """Return the prior-step health for a SMAC unit, falling back to current health."""
    if previous_units is not None and unit_id in previous_units:
        return float(getattr(previous_units[unit_id], "health", 0.0))
    return float(getattr(current_unit, "health", 0.0))


class _RewardUnitProxy:
    """Compatibility wrapper for SMAC reward functions generated by the LLM."""

    def __init__(self, unit, previous_health: float):
        self._unit = unit
        self.previous_health = float(previous_health)

    @property
    def is_alive(self) -> bool:
        return float(getattr(self._unit, "health", 0.0)) > 0.0

    @property
    def pos_x(self) -> float:
        return float(self._unit.pos.x)

    @property
    def pos_y(self) -> float:
        return float(self._unit.pos.y)

    def __getattr__(self, name):
        return getattr(self._unit, name)


class _RewardEnvProxy:
    """Expose the real SMAC env plus extra compatibility attributes for rewards."""

    def __init__(self, env):
        self._env = env
        previous_allies = getattr(env, "previous_ally_units", None)
        previous_enemies = getattr(env, "previous_enemy_units", None)
        self.agents = {
            unit_id: _RewardUnitProxy(
                unit=unit,
                previous_health=_previous_unit_health(previous_allies, unit_id, unit),
            )
            for unit_id, unit in getattr(env, "agents", {}).items()
        }
        self.enemies = {
            unit_id: _RewardUnitProxy(
                unit=unit,
                previous_health=_previous_unit_health(previous_enemies, unit_id, unit),
            )
            for unit_id, unit in getattr(env, "enemies", {}).items()
        }
        self.previous_ally_units = getattr(env, "previous_ally_units", {})
        self.previous_enemy_units = getattr(env, "previous_enemy_units", {})

    def __getattr__(self, name):
        return getattr(self._env, name)


class _PatchedSmacv2EnvFactory:
    """Create SMACv2 envs and apply the popgen training patch at runtime."""

    def __init__(
        self,
        task_config: dict,
        *,
        seed: int | None,
        device,
        group_map: dict[str, list[str]] | None = None,
    ):
        self._task_config = deepcopy(task_config)
        self._seed = seed
        self._device = device
        self._group_map = deepcopy(group_map) if group_map is not None else None

    def __call__(self):
        env = SMACv2Env(
            categorical_actions=True,
            seed=self._seed,
            device=self._device,
            **deepcopy(self._task_config),
        )
        return _patch_smacv2_env(env, group_map=self._group_map)


class _PatchedSmacv2GetEnvFun:
    """Picklable replacement for ``task.get_env_fun`` during popgen training."""

    def __init__(self, task_config: dict, group_map: dict[str, list[str]] | None = None):
        self._task_config = deepcopy(task_config)
        self._group_map = deepcopy(group_map) if group_map is not None else None

    def __call__(self, num_envs, continuous_actions, seed, device):
        del num_envs, continuous_actions
        return _PatchedSmacv2EnvFactory(
            self._task_config,
            seed=seed,
            device=device,
            group_map=self._group_map,
        )


def _agent_indices_for_group_map(group_map: dict[str, list[str]]) -> dict[str, list[int]]:
    return {
        group: [int(agent_name) for agent_name in agent_names]
        for group, agent_names in group_map.items()
    }


def _slice_group_spec(agent_spec, agent_indices: list[int]):
    if not agent_indices:
        raise ValueError("Expected at least one agent index per group.")
    start = agent_indices[0]
    stop = start + len(agent_indices)
    if agent_indices != list(range(start, stop)):
        raise ValueError(
            f"SMACv2 group split expects contiguous agent indices, got {agent_indices}."
        )
    return agent_spec[start:stop]


def _build_grouped_observation_spec(raw_observation_spec, group_map: dict[str, list[str]], device):
    agents_spec = raw_observation_spec["agents"]
    grouped_spec = {
        "state": raw_observation_spec["state"],
        "info": raw_observation_spec["info"],
    }

    for group, agent_indices in _agent_indices_for_group_map(group_map).items():
        n_agents = len(agent_indices)
        grouped_spec[group] = Composite(
            {
                "observation": _slice_group_spec(agents_spec["observation"], agent_indices),
                "action_mask": _slice_group_spec(agents_spec["action_mask"], agent_indices),
            },
            shape=(n_agents,),
        )

    return Composite(grouped_spec, shape=raw_observation_spec.shape, device=device)


def _build_grouped_action_spec(raw_action_spec, group_map: dict[str, list[str]], device):
    action_spec = raw_action_spec["agents"]["action"]
    grouped_spec = {}

    for group, agent_indices in _agent_indices_for_group_map(group_map).items():
        n_agents = len(agent_indices)
        grouped_spec[group] = Composite(
            {"action": _slice_group_spec(action_spec, agent_indices)},
            shape=(n_agents,),
        )

    return Composite(grouped_spec, shape=raw_action_spec.shape, device=device)


def _build_grouped_reward_spec(raw_reward_spec, group_map: dict[str, list[str]], device):
    reward_spec = raw_reward_spec["reward"]
    reward_tail_shape = reward_spec.shape[-1:] if len(reward_spec.shape) else torch.Size([1])
    grouped_spec = {}

    for group, agent_names in group_map.items():
        n_agents = len(agent_names)
        grouped_spec[group] = Composite(
            {
                "reward": Unbounded(
                    shape=torch.Size([n_agents, *reward_tail_shape]),
                    device=device,
                    dtype=reward_spec.dtype,
                )
            },
            shape=(n_agents,),
        )

    return Composite(grouped_spec, shape=raw_reward_spec.shape, device=device)


def _encode_smac_info(raw_observation_spec, info: dict) -> TensorDict:
    info_td = raw_observation_spec["info"].encode(info)
    actual_keys = info_td.keys()
    for expected_key, spec in raw_observation_spec["info"].items():
        if expected_key not in actual_keys:
            info_td[expected_key] = spec.zero()
    return info_td


def _update_smacv2_action_mask(env) -> torch.Tensor:
    mask = torch.tensor(
        env.get_avail_actions(),
        dtype=torch.bool,
        device=env.device,
    )

    group_indices = getattr(env, "_popgen_group_indices", None)
    if group_indices:
        for group, indices in group_indices.items():
            index = torch.tensor(indices, device=mask.device, dtype=torch.long)
            env.full_action_spec[group, "action"].update_mask(mask.index_select(-2, index))
    else:
        env.full_action_spec[env.action_key].update_mask(mask)

    return mask


def _split_grouped_agent_data(
    obs: torch.Tensor,
    action_mask: torch.Tensor,
    group_indices: dict[str, list[int]],
    *,
    device,
    rewards: torch.Tensor | None = None,
) -> dict[str, TensorDict]:
    grouped_data: dict[str, TensorDict] = {}

    for group, indices in group_indices.items():
        index = torch.tensor(indices, device=obs.device, dtype=torch.long)
        source = {
            "observation": obs.index_select(-2, index),
            "action_mask": action_mask.index_select(-2, index),
        }
        if rewards is not None:
            source["reward"] = rewards.index_select(-2, index)
        grouped_data[group] = TensorDict(
            source,
            batch_size=(len(indices),),
            device=device,
        )

    return grouped_data


def _merge_group_actions(tensordict: TensorDictBase, group_map: dict[str, list[str]], *, device) -> torch.Tensor:
    merged_action = None
    total_agents = sum(len(agent_names) for agent_names in group_map.values())

    for group, indices in _agent_indices_for_group_map(group_map).items():
        group_action = tensordict.get((group, "action"))
        if group_action is None:
            raise KeyError(f"Missing action for group '{group}'.")

        if len(indices) == 1 and group_action.dim() > 1 and group_action.shape[-1] == 1:
            group_action = group_action.squeeze(-1)
        if group_action.dim() == 0:
            group_action = group_action.unsqueeze(0)
        if group_action.shape[-1] != len(indices):
            raise ValueError(
                f"Group '{group}' action has shape {tuple(group_action.shape)}, "
                f"expected last dim {len(indices)}."
            )

        if merged_action is None:
            merged_action = torch.empty(
                (*group_action.shape[:-1], total_agents),
                dtype=group_action.dtype,
                device=device,
            )

        for local_idx, agent_idx in enumerate(indices):
            merged_action[..., agent_idx] = group_action[..., local_idx]

    if merged_action is None:
        raise ValueError("Could not merge actions because no group actions were found.")
    return merged_action


def _compute_smac_agent_rewards(
    env,
    reward_fn: Callable | None,
    fallback_reward: torch.Tensor,
) -> torch.Tensor:
    if reward_fn is None:
        reward_value = float(fallback_reward.mean().item())
        return torch.full(
            (env.n_agents, 1),
            reward_value,
            device=env.device,
            dtype=torch.float32,
        )

    reward_env = _RewardEnvProxy(env._env)
    rewards = []
    for agent_idx in range(env.n_agents):
        reward_value = reward_fn(reward_env, agent_idx)
        if isinstance(reward_value, torch.Tensor):
            reward_value = reward_value.item()
        rewards.append(float(reward_value))
    return torch.tensor(rewards, device=env.device, dtype=torch.float32).unsqueeze(-1)


class _GroupedSmacv2ObservationSpec:
    """Return per-group observations from a grouped SMACv2 env."""

    def __call__(self, env) -> Composite:
        observation_spec = env.observation_spec.clone()
        if "info" in observation_spec.keys():
            del observation_spec["info"]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        for group in list(env.group_map.keys()):
            if (group, "action_mask") in observation_spec.keys(True, True):
                del observation_spec[(group, "action_mask")]
        return observation_spec


class _GroupedSmacv2ActionMaskSpec:
    """Return per-group action masks from a grouped SMACv2 env."""

    def __call__(self, env) -> Composite:
        action_mask_spec = env.observation_spec.clone()
        if "info" in action_mask_spec.keys():
            del action_mask_spec["info"]
        if "state" in action_mask_spec.keys():
            del action_mask_spec["state"]
        for group in list(env.group_map.keys()):
            if (group, "observation") in action_mask_spec.keys(True, True):
                del action_mask_spec[(group, "observation")]
        return action_mask_spec


class _GroupedSmacv2StateSpec:
    """Return the global state spec from a grouped SMACv2 env."""

    def __call__(self, env) -> Composite:
        return Composite({"state": env.observation_spec["state"]}, shape=env.batch_size, device=env.device)


class _GroupedSmacv2InfoSpec:
    """Return the info spec from a grouped SMACv2 env."""

    def __call__(self, env) -> Composite:
        return Composite({"info": env.observation_spec["info"]}, shape=env.batch_size, device=env.device)


class _GroupedSmacv2ActionSpec:
    """Return grouped action specs from a grouped SMACv2 env."""

    def __call__(self, env) -> Composite:
        return env.full_action_spec


def _patch_smacv2_env(env, *, group_map: dict[str, list[str]] | None = None):
    reward_fn = _ACTIVE_SMACV2_REWARD_FN
    if reward_fn is not None and hasattr(reward_fn, "__globals__"):
        reward_fn.__globals__["torch"] = torch

    if group_map is None and reward_fn is None:
        return env

    if group_map is not None:
        env._popgen_group_map = deepcopy(group_map)
        env._popgen_group_indices = _agent_indices_for_group_map(group_map)
        env._popgen_raw_observation_spec = env.observation_spec.clone()
        env._popgen_raw_action_spec = env.full_action_spec.clone()
        env._popgen_raw_reward_spec = env.full_reward_spec.clone()
        env.group_map = deepcopy(group_map)
        env.observation_spec = _build_grouped_observation_spec(
            env._popgen_raw_observation_spec,
            group_map,
            env.device,
        )
        env.full_action_spec = _build_grouped_action_spec(
            env._popgen_raw_action_spec,
            group_map,
            env.device,
        )
        env.full_reward_spec = _build_grouped_reward_spec(
            env._popgen_raw_reward_spec,
            group_map,
            env.device,
        )

        def custom_reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
            del tensordict, kwargs
            obs, state = self._env.reset()
            obs = self._to_tensor(obs)
            state = self._to_tensor(state)
            info = self._popgen_raw_observation_spec["info"].zero()
            mask = _update_smacv2_action_mask(self)

            source = {"state": state, "info": info}
            source.update(
                _split_grouped_agent_data(
                    obs,
                    mask,
                    self._popgen_group_indices,
                    device=self.device,
                )
            )
            return TensorDict(source=source, batch_size=(), device=self.device)

        def custom_step(self, tensordict: TensorDictBase) -> TensorDictBase:
            action = _merge_group_actions(
                tensordict,
                self._popgen_group_map,
                device=self.device,
            )
            action_np = action.detach().cpu().numpy()

            try:
                reward, done, info = self._env.step(action_np)
            except AssertionError as err:
                if re.match(r"Agent . cannot perform action .", str(err)):
                    raise ACTION_MASK_ERROR
                raise err

            obs = self.get_obs()
            state = self.get_state()
            info_td = _encode_smac_info(self._popgen_raw_observation_spec, info)
            reward_td = torch.tensor(
                reward,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(-1)
            done_td = torch.tensor(
                done,
                device=self.device,
                dtype=torch.bool,
            ).unsqueeze(-1)
            mask = _update_smacv2_action_mask(self)
            agent_rewards = _compute_smac_agent_rewards(
                self,
                reward_fn,
                reward_td,
            )

            source = {
                "state": state,
                "info": info_td,
                "done": done_td,
                "terminated": done_td.clone(),
            }
            source.update(
                _split_grouped_agent_data(
                    obs,
                    mask,
                    self._popgen_group_indices,
                    device=self.device,
                    rewards=agent_rewards,
                )
            )
            return TensorDict(source=source, batch_size=(), device=self.device)

        env._reset = custom_reset.__get__(env, env.__class__)
        env._step = custom_step.__get__(env, env.__class__)
        return env

    original_step = env._step

    def reward_only_step(self, tensordict: TensorDictBase) -> TensorDictBase:
        out_td = original_step(tensordict)
        agent_rewards = _compute_smac_agent_rewards(
            self,
            reward_fn,
            out_td.get("reward"),
        )
        out_td["agents", "reward"] = agent_rewards
        return out_td

    env._step = reward_only_step.__get__(env, env.__class__)
    return env


class _RemappedActionSpec:
    """Callable that remaps SMACv2's flat 'agents' action spec to ego/teammate groups.

    SMACv2's ``env.full_action_spec`` is structured as::

        Composite({'agents': Composite({'action': spec}, shape=(n_agents,))})

    BenchMARL's ``_check_specs`` requires ``action_spec[group]`` to exist for
    every key in ``group_map``.  This callable rebuilds the spec so that each
    split group ('ego', 'teammate', …) gets its own per-agent Composite whose
    'action' sub-spec has batch shape ``(len(group_agents),)``.
    """

    def __init__(self, group_map: dict):
        self._group_map = group_map  # e.g. {'ego': ['0'], 'teammate': ['1']}

    def __call__(self, env) -> "Composite":
        from torchrl.data import Composite
        # action spec shape: (n_agents,) — slice per group
        action_spec = env.full_action_spec["agents"]["action"]
        offset = 0
        result = {}
        for group, agent_names in self._group_map.items():
            n = len(agent_names)
            group_action = action_spec[offset : offset + n]
            result[group] = Composite({"action": group_action}, shape=(n,))
            offset += n
        return Composite(result)


class _RemappedObservationSpec:
    """Expose per-group observation specs for a split SMACv2 team."""

    def __init__(self, group_map: dict):
        self._group_map = group_map

    def __call__(self, env) -> "Composite":
        from torchrl.data import Composite

        agents_spec = env.observation_spec["agents"]
        observation_spec = agents_spec["observation"]
        offset = 0
        result = {}
        for group, agent_names in self._group_map.items():
            n = len(agent_names)
            group_observation = observation_spec[offset : offset + n]
            result[group] = Composite({"observation": group_observation}, shape=(n,))
            offset += n
        return Composite(result)


class _RemappedActionMaskSpec:
    """Expose per-group action-mask specs for a split SMACv2 team."""

    def __init__(self, group_map: dict):
        self._group_map = group_map

    def __call__(self, env) -> "Composite":
        from torchrl.data import Composite

        agents_spec = env.observation_spec["agents"]
        action_mask_spec = agents_spec["action_mask"]
        offset = 0
        result = {}
        for group, agent_names in self._group_map.items():
            n = len(agent_names)
            group_mask = action_mask_spec[offset : offset + n]
            result[group] = Composite({"action_mask": group_mask}, shape=(n,))
            offset += n
        return Composite(result)


def _build_smacv2_train_group_map(
    task_config: dict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build SMACv2 training group map separating ego from scripted teammates.

    Returns:
        A ``(group_map, trainable_group)`` tuple where:
        - *group_map* – ``{"ego": ["0"], "teammate": ["1", ..., "n-1"]}``
        - *trainable_group* – ``{"ego": ["0"]}``
    """
    n_units = task_config.get("capability_config", {}).get("n_units")
    if n_units is None:
        raise ValueError(
            "Could not infer number of agents from SMACv2 task config "
            "(expected capability_config.n_units)."
        )
    if n_units < 2:
        raise ValueError(
            f"Need at least 2 agents to form ego/teammate split; got {n_units}."
        )
    agent_names = [str(i) for i in range(n_units)]
    group_map = {"ego": [agent_names[0]], "teammate": agent_names[1:]}
    trainable_group = {"ego": [agent_names[0]]}
    return group_map, trainable_group


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
    """Extract canonical rollout state from a TensorDict step.

    Returns ``{"ego_team": {...}, "opponent_team": {...}}``.
    Each agent entry contains ``state``, ``action``, ``reward``, and
    optionally ``action_mask`` (when present in the TensorDict).
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

        try:
            mask = td[group, "action_mask"]
        except KeyError:
            mask = None

        obs_2d = obs.reshape(-1, obs.shape[-1])
        num_agents = obs_2d.shape[0]
        act_2d = act.reshape(num_agents, -1) if act is not None else None
        rew_2d = rew.reshape(num_agents, -1) if rew is not None else None
        mask_2d = mask.reshape(num_agents, -1) if mask is not None else None

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

            if mask_2d is not None:
                payload["action_mask"] = mask_2d[i].cpu().tolist()

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
    rollout_td,
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


def _attach_rewards_from_rollout(states: list[dict], rollout_td) -> None:
    if not states:
        return

    nested_keys = set(rollout_td.keys(include_nested=True))
    top_groups = [k for k in rollout_td.keys() if isinstance(k, str)]
    reward_groups = [g for g in top_groups if ("next", g, "reward") in nested_keys]
    num_steps = len(states)

    # SMACv2 default env stores a single global reward at ("next", "reward")
    # rather than per-group. Fall back to broadcasting it to all agents.
    global_reward_t = None
    if not reward_groups and ("next", "reward") in nested_keys:
        global_reward_t = rollout_td["next", "reward"]  # shape [T] or [T, 1]

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

        if global_reward_t is not None:
            r = float(global_reward_t[step_idx].reshape(-1)[0].item())
            for name in ego_names:
                step["ego_team"][name]["reward"] = r
            for name in opp_names:
                step["opponent_team"][name]["reward"] = r
            continue

        ego_vals: list[float] = []
        opp_vals: list[float] = []
        for g in reward_groups:
            vals = _extract_step_group_rewards(
                rollout_td=rollout_td, group=g, step_idx=step_idx, num_steps=num_steps
            )
            team = _group_to_team(group_name=g, all_groups=reward_groups)
            if team == "ego":
                ego_vals.extend(vals)
            else:
                opp_vals.extend(vals)

        for name, val in zip(ego_names, ego_vals):
            step["ego_team"][name]["reward"] = float(val)
        for name, val in zip(opp_names, opp_vals):
            step["opponent_team"][name]["reward"] = float(val)


def serialize_state_fn(rollout_data: list[dict]) -> list[dict]:
    """Parse the 50-dim SMACv2 observation into a labeled dict for LLM consumption."""
    N_MOVE, N_ENEMIES, N_ALLIES, N_PER_UNIT, N_OWN = 4, 4, 1, 8, 6

    def _parse_unit(feats: list) -> dict:
        return {
            "shootable": bool(feats[0] > 0),
            "distance": float(feats[1]),
            "relative_x": float(feats[2]),
            "relative_y": float(feats[3]),
            "health": float(feats[4]),
            "unit_type_bits": [float(feats[5]), float(feats[6]), float(feats[7])],
            "visible": bool(feats[0] > 0 or feats[4] > 0),
        }

    for step in rollout_data:
        for team_name in ("ego_team", "opponent_team"):
            for agent_state in step.get(team_name, {}).values():
                raw = agent_state.get("state")
                if not isinstance(raw, list) or len(raw) < N_MOVE + N_ENEMIES * N_PER_UNIT + N_ALLIES * N_PER_UNIT + N_OWN:
                    continue
                offset = 0
                move = raw[offset : offset + N_MOVE]; offset += N_MOVE
                enemies = []
                for i in range(N_ENEMIES):
                    e = _parse_unit(raw[offset : offset + N_PER_UNIT]); offset += N_PER_UNIT
                    e["slot"] = i; enemies.append(e)
                allies = []
                for i in range(N_ALLIES):
                    a = _parse_unit(raw[offset : offset + N_PER_UNIT]); offset += N_PER_UNIT
                    a["slot"] = i; allies.append(a)
                own = raw[offset : offset + N_OWN]
                visible_enemies = [e for e in enemies if e["visible"]]
                nearest = min(visible_enemies, key=lambda e: e["distance"], default=None)
                agent_state["serialized_state"] = {
                    "can_move": {
                        "north": bool(move[0] > 0), "south": bool(move[1] > 0),
                        "east": bool(move[2] > 0), "west": bool(move[3] > 0),
                    },
                    "own_health": float(own[0]),
                    "own_position": {"x": float(own[1]), "y": float(own[2])},
                    "enemies": enemies,
                    "allies": allies,
                    "visible_enemy_count": len(visible_enemies),
                    "shootable_enemies": [e for e in enemies if e["shootable"]],
                    "nearest_visible_enemy": nearest,
                }
    return rollout_data


def _raise_smacv2_rollout_seed_error(env, *, reason: str | None = None):
    message = (
        "SMACv2 rollout seed could not be applied to the provided env. "
        "Use an env created through this project's SMAC create_env / TorchRL "
        "SMACv2Env, or call rollout_fn with env=None."
    )
    if reason:
        message = f"{message} {reason}"
    raise RuntimeError(message)


def _clone_smacv2_transform_for_seed(transform, env):
    if transform is None:
        return None

    if hasattr(transform, "clone"):
        try:
            return transform.clone()
        except Exception:
            pass

    try:
        return deepcopy(transform)
    except Exception as err:
        _raise_smacv2_rollout_seed_error(
            env,
            reason=f"Failed to clone env transforms for a seeded temporary clone: {err}",
        )


def _clone_smacv2_env_for_seed(env, effective_seed: int):
    base_env = getattr(env, "base_env", None)
    transform = getattr(env, "transform", None)
    if base_env is not None and transform is not None:
        cloned_base_env = _clone_smacv2_env_for_seed(base_env, effective_seed)
        transform_clone = _clone_smacv2_transform_for_seed(transform, env)
        clone_kwargs = {
            "base_env": cloned_base_env,
            "transform": transform_clone,
            "cache_specs": getattr(env, "cache_specs", True),
            "auto_unwrap": False,
        }
        env_device = getattr(env, "device", None)
        if env_device is not None:
            clone_kwargs["device"] = env_device

        try:
            return type(env)(**clone_kwargs)
        except Exception:
            try:
                return type(env)(
                    cloned_base_env,
                    transform_clone,
                    cache_specs=getattr(env, "cache_specs", True),
                    auto_unwrap=False,
                    device=env_device,
                )
            except Exception as err:
                try:
                    cloned_base_env.close()
                except Exception:
                    pass
                _raise_smacv2_rollout_seed_error(
                    env,
                    reason=f"Failed to create a seeded temporary clone: {err}",
                )

    constructor_kwargs = getattr(env, "_constructor_kwargs", None)
    if not isinstance(constructor_kwargs, dict):
        _raise_smacv2_rollout_seed_error(
            env,
            reason=(
                f"Missing _constructor_kwargs on {type(env).__name__}, so a seeded "
                "temporary clone could not be created."
            ),
        )

    try:
        clone_kwargs = deepcopy(constructor_kwargs)
    except Exception:
        clone_kwargs = dict(constructor_kwargs)

    clone_kwargs["seed"] = effective_seed
    if "device" not in clone_kwargs:
        env_device = getattr(env, "device", None)
        if env_device is not None:
            clone_kwargs["device"] = env_device

    try:
        return type(env)(**clone_kwargs)
    except Exception as err:
        _raise_smacv2_rollout_seed_error(
            env,
            reason=f"Failed to create a seeded temporary clone: {err}",
        )


def _prepare_smacv2_rollout_env(
    env,
    *,
    seed: int | None = None,
    create_env_kwargs: dict | None = None,
):
    """Resolve the concrete rollout seed and return an env ready for rollout."""
    effective_seed = resolve_rollout_seed(seed)
    torch.manual_seed(effective_seed)

    if env is None:
        env_kwargs = dict(create_env_kwargs or {})
        env_kwargs["seed"] = effective_seed
        return create_env(**env_kwargs), effective_seed, True

    if hasattr(env, "set_seed"):
        try:
            env.set_seed(effective_seed)
            return env, effective_seed, False
        except NotImplementedError:
            pass

    rollout_env = _clone_smacv2_env_for_seed(env, effective_seed)
    # Re-apply ego/teammate grouping patch if the original env was patched.
    if hasattr(env, "_popgen_group_map"):
        rollout_env = _patch_smacv2_env(rollout_env, group_map=env._popgen_group_map)
    return rollout_env, effective_seed, True


def rollout_smacv2_env(
    policy,
    env,
    max_steps: int = 200,
    seed: int | None = None,
    return_info: bool = False,
    **kwargs,
):
    """Rollout one policy (or policy pair) and optionally return reward metadata.

    The rollout always resolves one concrete seed. If the provided SMAC env
    cannot be reseeded after construction, this function creates a temporary
    seeded clone for the rollout and leaves the caller-owned env untouched.

    Backward compatibility:
    - default ``return_info=False`` returns ``(states, images, error)``
    - ``return_info=True`` returns ``(states, images, error, rollout_info)``
    """
    policy = compose_rollout_policy(policy)
    policy.eval()

    states = []
    images = []
    create_env_kwargs = kwargs.pop("_create_env_kwargs", None)
    rollout_env = None
    close_rollout_env = False

    try:
        rollout_env, _, close_rollout_env = _prepare_smacv2_rollout_env(
            env,
            seed=seed,
            create_env_kwargs=create_env_kwargs,
        )

        def state_callback(env_inner, td):
            try:
                frame = env_inner.render(mode="rgb_array")
                if frame is not None:
                    import numpy as np
                    if isinstance(frame, np.ndarray):
                        images.append(frame)
            except Exception:
                pass
            states.append(_collect_step_state(td))

        rollout_sig = inspect.signature(rollout_env.rollout)
        supported_args = set(rollout_sig.parameters.keys())
        rollout_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}
        if "max_steps" in supported_args and "max_steps" not in rollout_kwargs:
            rollout_kwargs["max_steps"] = max_steps

        with torch.no_grad():
            rollout_td = rollout_env.rollout(
                policy=policy,
                callback=state_callback,
                auto_cast_to_device=True,
                break_when_all_done=True,
                trust_policy=True,
                **rollout_kwargs,
            )

        _attach_rewards_from_rollout(states, rollout_td)

        if return_info:
            rollout_info = extract_reward_rollout_info(rollout_td)
            return states, images, None, rollout_info

        return states, images, None
    finally:
        if close_rollout_env and rollout_env is not None and hasattr(rollout_env, "close"):
            try:
                rollout_env.close()
            except Exception:
                pass


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable with permissive false-like parsing.

    Any value other than ``0``, ``false``, ``no``, or ``off`` is treated as
    enabled. This keeps CLI usage simple for local experiments.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_SMACV2_QUIET_ABSL = _env_flag("POPGEN_SMACV2_QUIET_ABSL", True)
_SMACV2_SUPPRESS_SC2_STDIO = _env_flag("POPGEN_SMACV2_SUPPRESS_SC2_STDIO", True)
_SMACV2_SC2_PATCH_INSTALLED = False


def _configure_smacv2_logging(
    *,
    quiet_absl: bool | None = None,
    suppress_sc2_process_output: bool | None = None,
) -> None:
    """Configure SMACv2 startup logging for quieter training runs.

    This helper controls two independent noise sources:
    - ``quiet_absl`` lowers the ``absl`` / ``pysc2`` logger to ``WARNING``,
      which suppresses messages such as ``Launching SC2`` and websocket retry
      logs.
    - ``suppress_sc2_process_output`` monkeypatches
      ``pysc2.lib.sc_process.StarcraftProcess._launch`` so the child SC2
      binary writes ``stdout`` / ``stderr`` to ``DEVNULL`` by default. This
      suppresses plain process output such as ``Game has started.`` and stable
      id messages.

    The default values come from these environment variables:
    - ``POPGEN_SMACV2_QUIET_ABSL`` (default: enabled)
    - ``POPGEN_SMACV2_SUPPRESS_SC2_STDIO`` (default: enabled)

    The monkeypatch is installed once per Python process. Later calls can still
    toggle behavior because the patched launch function reads the current module
    globals each time it starts SC2.
    """

    global _SMACV2_QUIET_ABSL
    global _SMACV2_SUPPRESS_SC2_STDIO
    global _SMACV2_SC2_PATCH_INSTALLED

    if quiet_absl is not None:
        _SMACV2_QUIET_ABSL = quiet_absl
    if suppress_sc2_process_output is not None:
        _SMACV2_SUPPRESS_SC2_STDIO = suppress_sc2_process_output

    try:
        from absl import logging as absl_logging

        level = absl_logging.WARNING if _SMACV2_QUIET_ABSL else absl_logging.INFO
        absl_logging.set_verbosity(level)
        py_logging.getLogger("absl").setLevel(
            py_logging.WARNING if _SMACV2_QUIET_ABSL else py_logging.INFO
        )
    except Exception:
        pass

    if _SMACV2_SC2_PATCH_INSTALLED:
        return

    try:
        from pysc2.lib import sc_process
    except Exception:
        return

    original_launch = sc_process.StarcraftProcess._launch

    def _patched_launch(self, run_config, args, **kwargs):
        if not _SMACV2_SUPPRESS_SC2_STDIO:
            return original_launch(self, run_config, args, **kwargs)

        try:
            with sc_process.sw("popen"):
                return subprocess.Popen(
                    args,
                    cwd=run_config.cwd,
                    env=run_config.env,
                    stdout=kwargs.pop("stdout", subprocess.DEVNULL),
                    stderr=kwargs.pop("stderr", subprocess.STDOUT),
                    **kwargs,
                )
        except OSError:
            sc_process.logging.exception("Failed to launch")
            raise sc_process.SC2LaunchError("Failed to launch: %s" % args)

    sc_process.StarcraftProcess._launch = _patched_launch
    _SMACV2_SC2_PATCH_INSTALLED = True


def train_smacv2_env(
    reward_fn=None,
    output_dir=None,
    run_name: str | None = None,
    smacv2_task: str = "TERRAN_5_VS_5",
    task_name: str = "terran_5_vs_5",
    rl_cfg_dir=None,
    seed: int = 0,
    teammate_policy: list[nn.Module] | None = None,
    quiet_absl: bool | None = None,
    suppress_sc2_process_output: bool | None = None,
    **kwargs,
):
    """Train a BenchMARL SMACv2 experiment with optional reward overriding.

    Args:
        reward_fn: Optional callable ``(env, agent_id) -> float`` used to
            replace the per-agent reward emitted by the underlying SMACv2 env.
        output_dir: Directory where BenchMARL writes checkpoints and logs.
        run_name: W&B / experiment display name. Falls back to the basename of
            ``output_dir`` when omitted.
        smacv2_task: BenchMARL enum member name, for example
            ``"TERRAN_2_VS_4"``.
        task_name: YAML config stem under ``rl_cfg_dir``, for example
            ``"terran_2_vs_4"``.
        rl_cfg_dir: Base directory containing SMACv2 task and experiment YAMLs.
        seed: Random seed forwarded to BenchMARL.
        teammate_policy: Optional teammate policy or teammate population used
            for non-self-play training.
        quiet_absl: Optional override for ``POPGEN_SMACV2_QUIET_ABSL``.
            ``True`` suppresses pysc2 ``absl`` INFO logs.
        suppress_sc2_process_output: Optional override for
            ``POPGEN_SMACV2_SUPPRESS_SC2_STDIO``. ``True`` suppresses direct
            SC2 binary ``stdout`` / ``stderr`` output.
        **kwargs: Additional config overrides applied to the loaded
            experiment, algorithm, or model configs when matching attributes
            exist.

    Returns:
        The finished BenchMARL ``Experiment`` instance.
    """
    if not isinstance(run_name, str) or not run_name.strip():
        run_name = Path(output_dir).name if output_dir else "smacv2_run"

    run_name = run_name.strip()
    callbacks = []

    _configure_smacv2_logging(
        quiet_absl=quiet_absl,
        suppress_sc2_process_output=suppress_sc2_process_output,
    )

    base_cfg_dir = Path(rl_cfg_dir)
    global _ACTIVE_SMACV2_REWARD_FN
    previous_reward_fn = _ACTIVE_SMACV2_REWARD_FN
    _ACTIVE_SMACV2_REWARD_FN = reward_fn
    try:
        task_enum = getattr(Smacv2Task, smacv2_task)
        task = task_enum.get_from_yaml(path=str(base_cfg_dir / f"{task_name}.yaml"))
        experiment_config = ExperimentConfig.get_from_yaml(path=str(rl_cfg_dir / "base_experiment.yaml"))
        ippo_path = base_cfg_dir / "ippo.yaml"
        if ippo_path.exists():
            algorithm_config = IppoConfig.get_from_yaml(path=str(ippo_path))
        else:
            algorithm_config = MappoConfig.get_from_yaml(path=str(base_cfg_dir / "mappo.yaml"))

        model_config = MlpConfig.get_from_yaml(path=str(base_cfg_dir / "mlp.yaml"))
        critic_mlp_path = base_cfg_dir / "critic_mlp.yaml"
        critic_model_config = MlpConfig.get_from_yaml(path=str(critic_mlp_path)) if critic_mlp_path.exists() else model_config

        desired_group_map = None

        if teammate_policy is not None:
            group_map, trainable_group = _build_smacv2_train_group_map(task_config=task.config)
            desired_group_map = group_map
            task.get_env_fun = _PatchedSmacv2GetEnvFun(task.config, group_map=group_map)
            task.state_spec = _GroupedSmacv2StateSpec()
            task.info_spec = _GroupedSmacv2InfoSpec()
            task.observation_spec = _GroupedSmacv2ObservationSpec()
            task.action_mask_spec = _GroupedSmacv2ActionMaskSpec()
            task.action_spec = _GroupedSmacv2ActionSpec()

            teammate_flat = teammate_policy

            from popgen.agents.scripted_partner import TeammatePolicyCallback
            callbacks.append(
                TeammatePolicyCallback(
                    teammate_policies=teammate_flat,
                    trainable_group=trainable_group,
                )
            )
        elif reward_fn is not None:
            task.get_env_fun = _PatchedSmacv2GetEnvFun(task.config, group_map=None)

        for key, val in kwargs.items():
            applied = False
            for cfg_obj in (experiment_config, algorithm_config, model_config):
                if hasattr(cfg_obj, key):
                    setattr(cfg_obj, key, val)
                    applied = True
                    break
            if not applied:
                py_logging.warning(f"train_smacv2_env: unknown kwarg '{key}' — ignored.")

        experiment_config.save_folder = output_dir
        experiment_config.wandb_extra_kwargs = {"name": run_name, "tags": ["smacv2"]}

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
        _ACTIVE_SMACV2_REWARD_FN = previous_reward_fn

    return experiment


def rollout_fn(policy, env=None, **kwargs):
    """Rollout the policy in a SMACv2 environment.

    Args:
        policy: A trained policy module or list of policies [ego, teammate].
        env: An existing SMACv2 environment. When omitted, create_env-compatible
            kwargs are used to build one before rollout.
        **kwargs: Forwarded to rollout_smacv2_env (e.g. max_steps, seed). If
            ``env is None``, create_env-compatible kwargs such as ``task_name``,
            ``seed``, ``device``, ``quiet_absl``, and
            ``suppress_sc2_process_output`` are consumed during env creation.

    Returns:
        (states, images, error) or (states, images, error, rollout_info) when
        return_info=True is passed in kwargs.
    """
    rollout_kwargs = dict(kwargs)
    # Always build a fresh env with the correct ego/teammate grouping.
    # The env passed from load_reward_rollout_artifacts uses the unpatched
    # default group_map ("agents") and must not be reused.
    create_env_params = set(inspect.signature(create_env).parameters.keys())
    create_env_kwargs = {}
    for key in list(rollout_kwargs):
        if key in create_env_params:
            create_env_kwargs[key] = rollout_kwargs.pop(key)
    env = create_env(**create_env_kwargs)

    if "max_steps" not in rollout_kwargs:
        rollout_kwargs["max_steps"] = 200


    if isinstance(policy, list) and len(policy) > 1:
        policy = MultiPolicyWrapper(policies=policy)
    elif (
        not isinstance(policy, MultiPolicyWrapper)
        and hasattr(env, "group_map")
        and "teammate" in env.group_map
    ):
        # Ego-only policy with split ego/teammate env — add a scripted teammate.
        policy = MultiPolicyWrapper(policies=[policy, ScriptedPolicy(env)])

    return rollout_smacv2_env(policy=policy, env=env, **rollout_kwargs)


def train_fn(
    reward_fn: Callable | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    **kwargs,
):
    """Train agents in SMACv2 using BenchMARL with an optional custom reward function.

    Args:
        reward_fn: Callable ``(env, agent_id) -> float`` returning per-agent reward.
        output_dir: Directory to save results.
        run_name: Experiment name (required, non-empty).
        **kwargs: Forwarded to train_smacv2_env. Notable keys:
            - teammate_policy: list of nn.Module (or list of lists for population).
              If omitted, SMACv2 defaults to a single scripted teammate.
            - seed: int (default 0).
    """
    if not isinstance(run_name, str) or not run_name.strip():
        run_name = Path(output_dir).name if output_dir else "smacv2_run"

    teammate_policy = kwargs.pop("teammate_policy", None)
    kwargs.pop("run_name", None)

    if teammate_policy is None:
        temp_env = create_env(num_envs=1, seed=0, device="cpu")
        try:
            teammate_policy = [ScriptedPolicy(temp_env)]
        finally:
            temp_env.close()

    return train_smacv2_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        run_name=run_name.strip(),
        smacv2_task="TERRAN_2_VS_4",
        task_name="terran_2_vs_4",
        rl_cfg_dir=RL_CFG_DIR,
        teammate_policy=teammate_policy,
        **kwargs,
    )


# ---------------------------------------------------------------------------- #
#                              Scripted Policy                                 #
# ---------------------------------------------------------------------------- #


class ScriptedPolicy(nn.Module):
    """Heuristic policy: attack nearest enemy, otherwise move toward it.

    The policy parses SMACv2 enemy features from the local observation:
    - enemy_feats[:, 0]: attack-available (shootable now)
    - enemy_feats[:, 1]: normalized distance
    - enemy_feats[:, 2]: relative x
    - enemy_feats[:, 3]: relative y

    For each agent:
    1) find the closest visible enemy,
    2) attack it if the corresponding attack action is valid,
    3) otherwise move toward that enemy,
    4) if movement is blocked, fall back to the first valid masked action.
    """

    STOP = 1
    NORTH = 2
    SOUTH = 3
    EAST = 4
    WEST = 5

    _DEFAULT_ATTACK_START = 6
    _DEFAULT_MOVE_FEATS = 4
    _DEFAULT_ENEMY_FEATS = 8
    _DEFAULT_N_ENEMIES = 4

    def __init__(self, env=None):
        super().__init__()
        self._n_actions: int | None = None
        self._attack_action_start = self._DEFAULT_ATTACK_START
        self._move_feat_size = self._DEFAULT_MOVE_FEATS
        self._enemy_feat_size = self._DEFAULT_ENEMY_FEATS
        self._n_enemies: int | None = None

        if env is not None:
            try:
                self._n_actions = env.full_action_spec["agents", "action"].space.n
            except Exception:
                pass
            smac_env = getattr(env, "_env", env)
            try:
                self._attack_action_start = int(smac_env.n_actions_no_attack)
            except Exception:
                pass
            try:
                self._move_feat_size = int(smac_env.get_obs_move_feats_size())
            except Exception:
                pass
            try:
                enemy_shape = smac_env.get_obs_enemy_feats_size()
                if isinstance(enemy_shape, (tuple, list)) and len(enemy_shape) == 2:
                    n_enemies, n_enemy_feats = int(enemy_shape[0]), int(enemy_shape[1])
                    if n_enemies > 0:
                        self._n_enemies = n_enemies
                    if n_enemy_feats >= 4:
                        self._enemy_feat_size = n_enemy_feats
            except Exception:
                pass

        if self._n_enemies is None and self._n_actions is not None and self._n_actions > self._attack_action_start:
            self._n_enemies = self._n_actions - self._attack_action_start
        if self._n_enemies is None:
            self._n_enemies = self._DEFAULT_N_ENEMIES

        self.dummy_param = nn.Parameter(torch.empty(0))

    @staticmethod
    def _first_valid_action(mask_row: torch.Tensor | None) -> int:
        if mask_row is None:
            return 0
        valid = (mask_row > 0).nonzero(as_tuple=True)[0]
        return int(valid[0].item()) if valid.numel() > 0 else 0

    def _move_toward_enemy_action(self, rel_x: float, rel_y: float, mask_row: torch.Tensor | None) -> int:
        candidates: list[int] = []

        def push(action: int) -> None:
            if action not in candidates:
                candidates.append(action)

        # Match SMAC coordinate convention: positive relative y means "north".
        if abs(rel_x) > abs(rel_y):
            if rel_x > 0:
                push(self.EAST)
            elif rel_x < 0:
                push(self.WEST)
            if rel_y > 0:
                push(self.NORTH)
            elif rel_y < 0:
                push(self.SOUTH)
        else:
            if rel_y > 0:
                push(self.NORTH)
            elif rel_y < 0:
                push(self.SOUTH)
            if rel_x > 0:
                push(self.EAST)
            elif rel_x < 0:
                push(self.WEST)

        for action in (self.NORTH, self.SOUTH, self.EAST, self.WEST, self.STOP):
            push(action)

        for action in candidates:
            if mask_row is None:
                return action
            if action < mask_row.numel() and bool(mask_row[action].item()):
                return action

        return self._first_valid_action(mask_row)

    @staticmethod
    def _resolve_grouped_obs_key(obs: TensorDictBase) -> str:
        """Pick the observation group this scripted policy should control.

        In split-group training / rollout wrappers the teammate policy can
        receive the full ``ego`` + ``teammate`` TensorDict so that the trainable
        policy keeps its original grouped context. In that case the scripted
        partner should explicitly act on the teammate slice instead of calling
        :func:`find_observation_group`, which expects only one observation
        group to be present.
        """
        nested_keys = set(obs.keys(include_nested=True, leaves_only=True))
        if ("teammate", "observation") in nested_keys:
            return "teammate"
        return find_observation_group(obs)

    def forward(self, obs) -> "torch.Tensor | TensorDictBase":
        """Select nearest-enemy attack/chase actions for each observation.

        Args:
            obs: TensorDict (with ``action_mask``) or plain tensor.

        Returns:
            Action tensor or TensorDict with action key set.
        """
        action_mask = None
        action_key = None

        if isinstance(obs, TensorDictBase):
            # Flat layout: {"observation": ..., "action_mask": ...}
            if "action_mask" in obs.keys():
                action_mask = obs["action_mask"]
                obs_tensor, action_key = extract_observation_and_action_key(obs)
            else:
                # Grouped layout: {group: {"observation": ..., "action_mask": ...}}
                group = self._resolve_grouped_obs_key(obs)
                action_mask = obs[group, "action_mask"]
                obs_tensor = obs[group, "observation"]
                action_key = (group, "action")
        else:
            obs_tensor = obs

        flat_obs, leading_shape = flatten_obs(
            obs_tensor if isinstance(obs_tensor, torch.Tensor) else obs_tensor
        )
        batch_size = flat_obs.shape[0]
        device = flat_obs.device

        actions = torch.full((batch_size,), self.STOP, dtype=torch.long, device=device)
        flat_mask = None

        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            flat_mask = action_mask.reshape(batch_size, -1)

        for b in range(batch_size):
            mask_row = flat_mask[b] if flat_mask is not None else None
            n_enemies = self._n_enemies
            if mask_row is not None and mask_row.numel() > self._attack_action_start:
                n_attack_slots = int(mask_row.numel() - self._attack_action_start)
                if n_attack_slots > 0:
                    n_enemies = min(n_enemies, n_attack_slots)

            enemy_block_size = n_enemies * self._enemy_feat_size
            enemy_start = self._move_feat_size
            obs_row = flat_obs[b]

            if enemy_block_size <= 0 or obs_row.numel() < enemy_start + enemy_block_size:
                actions[b] = self._first_valid_action(mask_row) if mask_row is not None else self.STOP
                continue

            enemy_feats = obs_row[enemy_start: enemy_start + enemy_block_size].reshape(
                n_enemies, self._enemy_feat_size
            )

            shootable = enemy_feats[:, 0]
            dist = enemy_feats[:, 1]
            rel_x = enemy_feats[:, 2]
            rel_y = enemy_feats[:, 3]

            visible = (
                (shootable > 0)
                | (dist > 0)
                | (rel_x.abs() > 1e-6)
                | (rel_y.abs() > 1e-6)
            )
            if self._enemy_feat_size > 4:
                visible = visible | (enemy_feats[:, 4] > 0)

            visible_ids = visible.nonzero(as_tuple=True)[0]
            if visible_ids.numel() == 0:
                actions[b] = self._first_valid_action(mask_row) if mask_row is not None else self.STOP
                continue

            nearest = visible_ids[torch.argmin(dist[visible_ids])]
            nearest_id = int(nearest.item())
            attack_action = self._attack_action_start + nearest_id

            can_attack = bool(shootable[nearest].item() > 0)
            if mask_row is not None:
                can_attack = (
                    can_attack
                    and attack_action < mask_row.numel()
                    and bool(mask_row[attack_action].item())
                )

            if can_attack:
                actions[b] = attack_action
                continue

            actions[b] = self._move_toward_enemy_action(
                rel_x=float(rel_x[nearest].item()),
                rel_y=float(rel_y[nearest].item()),
                mask_row=mask_row,
            )

        if len(leading_shape) == 0:
            actions = actions.squeeze(0)
        else:
            actions = actions.reshape(*leading_shape)

        return format_action_output(obs, action_key, actions)


# ---------------------------------------------------------------------------- #
#                              Make env function                               #
# ---------------------------------------------------------------------------- #


def create_env(
    task_name: str = "terran_2_vs_4",
    num_envs: int = 1,
    seed=None,
    device: str = "cpu",
    quiet_absl: bool | None = None,
    suppress_sc2_process_output: bool | None = None,
):
    """Create a SMACv2 environment using BenchMARL's Smacv2Task.

    Args:
        task_name: Task name matching the yaml file in rl_cfg/ (e.g. "terran_2_vs_4").
        num_envs: Ignored for SMACv2 (non-vectorized); kept for API consistency.
        seed: Random seed.
        device: Torch device string.
        quiet_absl: Optional override for ``POPGEN_SMACV2_QUIET_ABSL``.
            ``True`` suppresses pysc2 INFO logs during SC2 launch / connect.
        suppress_sc2_process_output: Optional override for
            ``POPGEN_SMACV2_SUPPRESS_SC2_STDIO``. ``True`` suppresses direct
            SC2 binary stdout/stderr such as startup banner lines.

    Returns:
        An SMACv2Env instance (TorchRL EnvBase).
    """
    _configure_smacv2_logging(
        quiet_absl=quiet_absl,
        suppress_sc2_process_output=suppress_sc2_process_output,
    )
    task_enum_name = task_name.upper()
    task_enum = getattr(Smacv2Task, task_enum_name)
    task = task_enum.get_from_yaml(path=str(RL_CFG_DIR / f"{task_name}.yaml"))
    env = task.get_env_fun(
        num_envs=num_envs,
        continuous_actions=False,
        seed=seed,
        device=device,
    )()
    # Apply ego/teammate grouping so rollout envs match training format.
    group_map, _ = _build_smacv2_train_group_map(task.config)
    env = _patch_smacv2_env(env, group_map=group_map)
    return env
