import os
import random
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from benchmarl.environments import VmasTask
from pydantic import BaseModel, Field
from vmas import make_env
from vmas.scenarios import point_mass_rendezvous as point_mass_rendezvous_scenario

from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

# Task config overrides applied during self-play population training.
# Referenced by both population_generation.py (via _build_task) and train_fn.
SELFPLAY_SCENARIO_KWARGS: dict = {"shared_goal_shaping": True}

ENVIRONMENT_DESCRIPTION = """The game is called "Point Mass Rendezvous".
The objective of the game is for two agents to coordinate and navigate to the same landmark simultaneously in a 2D space.

There are 4 landmarks available, representing different potential solutions and two agents: Agent 0 (Red) and Agent 1 (Blue).
Each landmark is represented by its (x, y) coordinates.
The ego agent is represented as the red point mass. The other blue point mass is the teammate agent.
The agents must agree on which landmark to go to without explicit communication.

> **Observation Space**:
The observation space for each agent is a vector that includes the following:
- Index 0-1: `ego_agent_position` - A tuple (x, y) representing the position of ego agent.
- Index 2-3: `ego_agent_velocity` - A tuple (vx, vy) representing the velocity of ego agent.
- Index 4-11: `landmark_relative_positions` - A list of tuples [(x, y), ...] representing the relative positions of the 4 landmarks to the ego agent (landmark_position - ego_agent_position).
- Index 12-13: `other_agent_relative_position` - A tuple (x, y) representing the position of the other agent relative to the ego agent (other_agent_position - ego_agent_position).

> **Action Space**:
The agents can apply a force in the horizontal and vertical directions to move around the space.
The action is represented as a tuple (fx, fy) where:
- `fx`: Horizontal force component, where -1 means full force to the left and 1 means full force to the right.
- `fy`: Vertical force component, where -1 means full force downwards and 1 means full force upwards."""

REWARD_API_REFERENCE = """Reward function API (authoritative):
- Signature: `compute_reward(self, agent)`.
- In point-mass-rendezvous training, `self` is the VMAS scenario and `agent` is one VMAS agent.
- Accessible state:
  - `self.world`: world object.
  - `self.world.batch_dim`: number of parallel environments.
  - `self.world.agents`: list of agents.
  - `self.world.landmarks`: list of landmarks.
  - For agents/landmarks `e`: `e.state.pos` and (for agents) `e.state.vel` are tensors with leading dim `batch_dim`.
  - Radii are available via `e.shape.radius`.
- Return value:
  - Return a torch tensor of shape `(batch_dim,)` on the world device.
  - Keep the per-step reward within [-10, 10]. Values outside that range, and any NaN or inf, are clamped at runtime.
  - Keep dense per-step terms small, around |r| <= 1, and reserve larger values for rare events (order 1-5). The clamp bounds a single step, not the episode: a dense term of 1 per step already accumulates to hundreds over an episode, and returns that large make advantages explode, which drives the policy to NaN and kills the run. Aim for an episode return in the low tens.
  - Prefer normalized or tanh-shaped dense terms over raw distances or unbounded sums."""

REWARD_EXAMPLE = """API reference only -- this is the environment's own rendezvous objective, NOT a model for
the reward you are writing. Copy the tensor shapes and the way state is read; do not copy
what it rewards. A reward built on "both agents on the same landmark" is behavior-agnostic:
every behavior trained against it converges on the same rendezvous, whatever its
description said. Reward the described behavior instead (which landmark this agent favors,
whether it waits or leads, whether it follows the partner at all).

Simple sparse reward: +1 if both agents are on the same landmark at the same time step, otherwise 0.

```python
import torch

def compute_reward(self, agent):

    reward = torch.zeros(self.world.batch_dim, device=self.world.device)

    # You are designing reward for 'agent' in 'world'
    # You can access to list of agents and landmarks via world.agents and world.landmarks
    agents = self.world.agents
    landmarks = self.world.landmarks

    # You can access to agent's state via agent.state
    # agent.state.pos gives you the position of the agent in torch tensor shape (batch_dim, 2)
    # agent.state.vel gives you the velocity of the agent in torch tensor shape (batch_dim, 2)

    # You can access to landmark's state via landmark.state
    # landmark.state.pos gives you the position of the landmark in torch tensor shape (batch_dim, 2)

    # Example reward give +1 if both agents are on the same landmark
    for landmark in landmarks:
        # Check if all agents are on this landmark
        agents_on_landmark = torch.stack(
            [
                torch.linalg.vector_norm(
                    a.state.pos - landmark.state.pos, dim=1
                ) < a.shape.radius + landmark.shape.radius # shape.radius gives you the radius of the agent/landmark in scalar
                for a in agents
            ],
            dim=1,
        ).all(dim=1)
        reward[agents_on_landmark] += 1.0

    return reward
```"""

# ---------------------------------------------------------------------------- #
#                                 Action Type                                  #
# ---------------------------------------------------------------------------- #


class ActionType(BaseModel):
    """
    Action type for the point mass rendezvous environment.
    - fx: is the horizontal force component, where -1 means full force to the left and 1 means full force to the right.
    - fy: is the vertical force component, where -1 means full force downwards and 1 means full force upwards.

    Example: to move the agent to the right with a force of 1.0 and upwards with a force of 0.5, you would return `ActionType(fx=1.0, fy=0.5)`.
    """

    fx: float = Field(description="Horizontal force component (-1=left, 1=right)")
    fy: float = Field(description="Vertical force component (-1=down, 1=up)")

    @classmethod
    def default_action(cls):
        """Default action with zero force."""
        return cls(fx=0.0, fy=0.0)

    def to_tensor(self):
        """Convert action to a torch tensor of shape (2,)."""
        return torch.tensor([self.fx, self.fy], dtype=torch.float32)


# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #


def train_fn(reward_fn: Callable | None = None, output_dir: str | None = None, run_name: str | None = None, **kwargs):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward tensor.
        output_dir: Directory to save results.
    """
    teammate_policy = kwargs.pop("teammate_policy", None)
    is_self_play = teammate_policy is None  # capture intent before dummy creation

    # Auto-enable reward shaping for self-play population; disable for cooperator/FCP.
    # Explicit scenario_kwargs["shared_goal_shaping"] from the caller takes precedence.
    scenario_kwargs = kwargs.pop("scenario_kwargs", {})
    if "shared_goal_shaping" not in scenario_kwargs:
        scenario_kwargs["shared_goal_shaping"] = is_self_play

    if teammate_policy is None:
        # Use the training num_envs so scripted partner policy captures the right metadata.
        with open(RL_CFG_DIR / "rl_reward" / "base_experiment.yaml") as f:
            experiment_cfg = yaml.safe_load(f)
        num_envs = experiment_cfg.get("on_policy_n_envs_per_worker", 1)

        temp_env = create_env(num_envs=num_envs, seed=0, device="cpu")
        try:
            teammate_policy = [RandomLandmarkPolicy(temp_env)]
        finally:
            temp_env.close()

    task = VmasTask.POINT_MASS_RENDEZVOUS.get_from_yaml(path=str(RL_CFG_DIR / "point_mass_rendezvous.yaml"))
    group_map = build_group_map(task.config)

    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=point_mass_rendezvous_scenario,
        scenario_name="point_mass_rendezvous",
        vmas_task=VmasTask.POINT_MASS_RENDEZVOUS,
        rl_cfg_dir=RL_CFG_DIR,
        group_map=group_map,
        teammate_policy=teammate_policy,
        run_name=run_name,
        scenario_kwargs=scenario_kwargs,
        **kwargs,
    )


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy loaded from the checkpoint in save_dir.
    Uses rollout states extracted directly from tensordicts.
    """
    # if env is not defined, load default pm_rdv environment
    if env is None:
        env = create_env(**kwargs)

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 200  # Default max steps for point_mass_rendezvous

    return rollout_vmas_env(
        policy=policy,
        env=env,
        **kwargs,
    )


def serialize_state_fn(rollout_data: list[dict]) -> list[dict]:
    for step in rollout_data:
        for team_name in ("ego_team", "opponent_team"):
            for agent_state in step.get(team_name, {}).values():
                raw_state = agent_state.get("state")
                if isinstance(raw_state, list) and len(raw_state) >= 14:
                    ego_pos = [float(raw_state[0]), float(raw_state[1])]
                    ego_vel = [float(raw_state[2]), float(raw_state[3])]
                    other_rel = [float(raw_state[12]), float(raw_state[13])]
                    other_pos = [ego_pos[0] + other_rel[0], ego_pos[1] + other_rel[1]]
                    landmarks = []
                    nearest_landmark_slot = None
                    nearest_landmark_distance = None

                    for i in range(4):
                        rel = [float(raw_state[4 + 2 * i]), float(raw_state[5 + 2 * i])]
                        dist = float((rel[0] ** 2 + rel[1] ** 2) ** 0.5)
                        if nearest_landmark_distance is None or dist < nearest_landmark_distance:
                            nearest_landmark_distance = dist
                            nearest_landmark_slot = i
                        landmarks.append(
                            {
                                "slot": i,
                                "relative_pos": rel,
                                "pos": [ego_pos[0] + rel[0], ego_pos[1] + rel[1]],
                                "distance": dist,
                            }
                        )

                    agent_state["serialized_state"] = {
                        "ego_pos": ego_pos,
                        "ego_vel": ego_vel,
                        "other_agent_pos": other_pos,
                        "other_agent_distance": float((other_rel[0] ** 2 + other_rel[1] ** 2) ** 0.5),
                        "landmarks": landmarks,
                        "nearest_landmark_slot": nearest_landmark_slot,
                        "nearest_landmark_distance": nearest_landmark_distance,
                    }
    return rollout_data

# ---------------------------------------------------------------------------- #
#                              Make env function                               #
# ---------------------------------------------------------------------------- #


def build_group_map(task_cfg: dict) -> dict[str, list[str]]:
    """Build semantic group_map with 'ego' and 'teammate' from raw point-mass task config."""
    raw_group_map = task_cfg.get("group_map")
    if raw_group_map:
        seen: set[str] = set()
        agent_names: list[str] = []
        for grouped_agents in raw_group_map.values():
            for name in grouped_agents:
                if name not in seen:
                    seen.add(name)
                    agent_names.append(name)
    else:
        n_agents = task_cfg.get("n_agents")
        if n_agents is None:
            raise ValueError("Point-mass task config must define 'n_agents' or 'group_map'.")
        agent_names = [f"agent_{i}" for i in range(int(n_agents))]
    if not agent_names:
        raise ValueError("Point-mass group_map must contain at least one controlled agent.")
    group_map = {"ego": [agent_names[0]]}
    if len(agent_names) > 1:
        group_map["teammate"] = agent_names[1:]
    return group_map


def create_env(num_envs=1, continuous_actions=True, seed=None, device="cpu", **kwargs):
    """
    Creates the point mass rendezvous environment using VMAS.

    Args:
        kwargs: Keyword arguments to pass to the environment creation function. Can include:
            num_envs, device, continuous_actions, max_steps, seed, dict_spaces, multidiscrete_actions, clamp_actions, grad_enabled, terminated_truncated, wrapper_kwargs, etc.

    Returns:
        An instance of the point mass rendezvous environment.
    """
    task = VmasTask.POINT_MASS_RENDEZVOUS.get_from_yaml(path=str(RL_CFG_DIR / "point_mass_rendezvous.yaml"))
    task.config["group_map"] = build_group_map(task.config)
    return task.get_env_fun(
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
    )()

# ---------------------------------------------------------------------------- #
#                              Scripted Policy                                 #
# ---------------------------------------------------------------------------- #


def _extract_policy_inputs(obs: torch.Tensor, num_landmarks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate obs tensor and parse landmark and teammate relative positions."""
    if not isinstance(obs, torch.Tensor):
        raise TypeError(f"PointMass ScriptedPolicy expected torch.Tensor observation, got {type(obs).__name__}.")
    if obs.dim() != 2:
        raise ValueError(f"PointMass ScriptedPolicy expects observation tensor with shape (batch, obs_dim), got {tuple(obs.shape)}.")
    batch_size = obs.shape[0]
    landmark_end = 4 + num_landmarks * 2
    if obs.shape[-1] < landmark_end + 2:
        raise ValueError(
            f"PointMass observation too short for {num_landmarks} landmarks: "
            f"obs_dim={obs.shape[-1]}, expected at least {landmark_end + 2}."
        )
    landmark_relative_positions = obs[:, 4:landmark_end].view(batch_size, num_landmarks, 2)
    other_agent_relative_position = obs[:, landmark_end:landmark_end + 2]
    return landmark_relative_positions, other_agent_relative_position


def _normalize_action(direction: torch.Tensor) -> torch.Tensor:
    return (direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8)).clamp(-1, 1)


class _PointMassScriptedPolicyBase(torch.nn.Module):
    """Shared utilities for scripted point-mass teammate policies."""

    def __init__(self, env=None, num_landmarks=4, num_agents=2):
        super().__init__()
        self.num_envs = env.num_envs if env is not None else 1
        self.num_landmarks = len(env.world.landmarks) if env is not None else num_landmarks
        self.num_agents = len(env.world.agents) if env is not None else num_agents
        # nn.Module requires at least one parameter for some execution paths.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))


class RandomLandmarkPolicy(_PointMassScriptedPolicyBase):
    """Random-commit partner: picks one landmark at reset and keeps pursuing it."""

    def __init__(self, env=None, num_landmarks=4, num_agents=2):
        super().__init__(env=env, num_landmarks=num_landmarks, num_agents=num_agents)
        self._rng = random.Random(int.from_bytes(os.urandom(4), "little"))
        self.target_landmark_idx = torch.tensor(
            [self._rng.randrange(self.num_landmarks) for _ in range(self.num_envs)],
            dtype=torch.long,
        )

    def _ensure_targets(self, batch_size: int) -> None:
        if self.target_landmark_idx.numel() >= batch_size:
            return
        extra = torch.tensor(
            [self._rng.randrange(self.num_landmarks) for _ in range(batch_size - self.target_landmark_idx.numel())],
            dtype=torch.long,
        )
        self.target_landmark_idx = torch.cat([self.target_landmark_idx, extra], dim=0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        landmark_relative_positions, _other_agent_relative_position = _extract_policy_inputs(obs, self.num_landmarks)
        batch_size = obs.shape[0]
        self._ensure_targets(batch_size)
        target_idx = self.target_landmark_idx[:batch_size].to(obs.device)
        direction_to_landmark = landmark_relative_positions[torch.arange(batch_size, device=obs.device), target_idx]
        return _normalize_action(direction_to_landmark)


class ClosestLandmarkPolicy(_PointMassScriptedPolicyBase):
    """Greedy partner: always moves toward the currently closest landmark."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        landmark_relative_positions, _other_agent_relative_position = _extract_policy_inputs(obs, self.num_landmarks)
        batch_size = obs.shape[0]
        distances = torch.linalg.norm(landmark_relative_positions, dim=-1)
        target_idx = torch.argmin(distances, dim=-1)
        direction_to_landmark = landmark_relative_positions[torch.arange(batch_size, device=obs.device), target_idx]
        return _normalize_action(direction_to_landmark)


class TeammateBiasedLandmarkPolicy(_PointMassScriptedPolicyBase):
    """Supportive partner: targets the landmark closest to the teammate."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        landmark_relative_positions, other_agent_relative_position = _extract_policy_inputs(obs, self.num_landmarks)
        batch_size = obs.shape[0]
        # teammate->landmark = (landmark - ego) - (teammate - ego)
        teammate_to_landmarks = landmark_relative_positions - other_agent_relative_position.unsqueeze(1)
        teammate_distances = torch.linalg.norm(teammate_to_landmarks, dim=-1)
        target_idx = torch.argmin(teammate_distances, dim=-1)
        direction_to_landmark = landmark_relative_positions[torch.arange(batch_size, device=obs.device), target_idx]
        return _normalize_action(direction_to_landmark)


class RegroupThenClosestLandmarkPolicy(_PointMassScriptedPolicyBase):
    """Two-stage partner: regroup with teammate first, then go to nearest landmark."""

    def __init__(self, env=None, num_landmarks=4, num_agents=2, regroup_distance: float = 0.75):
        super().__init__(env=env, num_landmarks=num_landmarks, num_agents=num_agents)
        self.regroup_distance = float(regroup_distance)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        landmark_relative_positions, other_agent_relative_position = _extract_policy_inputs(obs, self.num_landmarks)
        batch_size = obs.shape[0]
        landmark_distances = torch.linalg.norm(landmark_relative_positions, dim=-1)
        target_idx = torch.argmin(landmark_distances, dim=-1)
        nearest_landmark_direction = landmark_relative_positions[
            torch.arange(batch_size, device=obs.device),
            target_idx,
        ]
        teammate_distance = torch.linalg.norm(other_agent_relative_position, dim=-1)
        should_regroup = teammate_distance > self.regroup_distance
        direction = torch.where(
            should_regroup.unsqueeze(-1),
            other_agent_relative_position,
            nearest_landmark_direction,
        )
        return _normalize_action(direction)


class FarthestLandmarkPolicy(_PointMassScriptedPolicyBase):
    """Anti-greedy partner: commits to the farthest landmark on the first step and pursues it persistently."""

    def __init__(self, env=None, num_landmarks=4, num_agents=2):
        super().__init__(env=env, num_landmarks=num_landmarks, num_agents=num_agents)
        self.target_landmark_idx: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        landmark_relative_positions, _other_agent_relative_position = _extract_policy_inputs(obs, self.num_landmarks)
        batch_size = obs.shape[0]

        if self.target_landmark_idx is None:
            self.target_landmark_idx = obs.new_full((batch_size,), -1, dtype=torch.long)
        elif self.target_landmark_idx.shape[0] < batch_size:
            extra = obs.new_full((batch_size - self.target_landmark_idx.shape[0],), -1, dtype=torch.long)
            self.target_landmark_idx = torch.cat([self.target_landmark_idx, extra], dim=0)

        uncommitted = self.target_landmark_idx[:batch_size] < 0
        if uncommitted.any():
            distances = torch.linalg.norm(landmark_relative_positions, dim=-1)
            farthest_idx = torch.argmax(distances, dim=-1)
            self.target_landmark_idx[:batch_size] = torch.where(uncommitted, farthest_idx, self.target_landmark_idx[:batch_size])

        target_idx = self.target_landmark_idx[:batch_size].to(obs.device)
        direction_to_landmark = landmark_relative_positions[torch.arange(batch_size, device=obs.device), target_idx]
        return _normalize_action(direction_to_landmark)


class RegroupThenDivergePolicy(_PointMassScriptedPolicyBase):
    """Approaches the teammate until nearby, then commits to the landmark farthest from them."""

    def __init__(self, env=None, num_landmarks=4, num_agents=2, regroup_threshold: float = 0.3):
        super().__init__(env=env, num_landmarks=num_landmarks, num_agents=num_agents)
        self.regroup_threshold = float(regroup_threshold)
        self._committed_idx: torch.Tensor | None = None  # -1 = still regrouping

    def _ensure_state(self, batch_size: int, device: torch.device) -> None:
        if self._committed_idx is None:
            self._committed_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        elif self._committed_idx.shape[0] < batch_size:
            extra = batch_size - self._committed_idx.shape[0]
            self._committed_idx = torch.cat([self._committed_idx, torch.full((extra,), -1, dtype=torch.long, device=device)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        landmark_relative_positions, other_agent_relative_position = _extract_policy_inputs(obs, self.num_landmarks)
        batch_size = obs.shape[0]
        self._ensure_state(batch_size, obs.device)

        committed = self._committed_idx[:batch_size]

        # Once close enough to teammate, commit to the landmark farthest from them.
        teammate_dist = torch.linalg.norm(other_agent_relative_position, dim=-1)
        should_commit = (teammate_dist < self.regroup_threshold) & (committed < 0)
        if should_commit.any():
            teammate_to_landmarks = landmark_relative_positions - other_agent_relative_position.unsqueeze(1)
            teammate_distances = torch.linalg.norm(teammate_to_landmarks, dim=-1)
            farthest_from_teammate = torch.argmax(teammate_distances, dim=-1)
            committed = torch.where(should_commit, farthest_from_teammate, committed)
            self._committed_idx[:batch_size] = committed

        # Phase 1: move toward teammate. Phase 2: move toward committed landmark.
        is_committed = committed >= 0
        target_direction = landmark_relative_positions[torch.arange(batch_size, device=obs.device), committed.clamp(min=0)]
        action = torch.where(
            is_committed.unsqueeze(-1),
            _normalize_action(target_direction),
            _normalize_action(other_agent_relative_position),
        )
        return action


# Use random landmark policy as the default scripted policy.
ScriptedPolicy = RandomLandmarkPolicy

EVALUATION_BEHAVIOR_SET: list[dict[str, Any]] = [
    {
        "name": "closest_landmark_greedy",
        "description": "Greedily selects the nearest landmark at each step.",
        "policy_cls": ClosestLandmarkPolicy,
    },
    {
        "name": "farthest_landmark_commitment",
        "description": "Always moves toward the currently farthest landmark.",
        "policy_cls": FarthestLandmarkPolicy,
    },
    {
        "name": "teammate_biased_landmark",
        "description": "Prioritizes the landmark closest to the teammate.",
        "policy_cls": TeammateBiasedLandmarkPolicy,
    },
    {
        "name": "regroup_then_diverge",
        "description": "Approaches the teammate until nearby, then commits to the landmark farthest from them.",
        "policy_cls": RegroupThenDivergePolicy,
    },
]

def get_evaluation_scripted_policies() -> list[Callable[[Any], torch.nn.Module]]:
    """Return scripted policy factories for evaluation behavior sweeps."""
    return [entry["policy_cls"] for entry in EVALUATION_BEHAVIOR_SET]
