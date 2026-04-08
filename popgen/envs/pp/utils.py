from pathlib import Path

import torch
import yaml
from benchmarl.environments import VmasTask
from pydantic import BaseModel, Field
from torchrl.envs.libs.vmas import VmasEnv

from . import semdiv_simple_tag as simple_tag_scenario
from popgen.envs.vmas_common import default_trainable_group, rollout_vmas_env, train_vmas_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

RL_CFG_DIR = Path(__file__).parent / "rl_cfg"
EGO_AGENT_NAME = "adversary_0"
EGO_AGENT_DARK_RED = (0.55, 0.0, 0.0)

ENVIRONMENT_DESCRIPTION = """The environment is an MPE predator-prey tag game (VMAS `simple_tag`).

There are two teams:
- Predators (adversaries, red): try to tag prey by collision.
- Prey (good agents, green): try to avoid being tagged.

Default setup here is 2v5:
- Predators: `adversary_0`, `adversary_1`
- Prey: `agent_0`..`agent_4` where:
  - `agent_0`, `agent_1` are stags (require both predators simultaneously)
  - `agent_2`, `agent_3`, `agent_4` are rabbits (single-predator capture)

The controlled ego agent is `adversary_0`.

================================================================================
CRITICAL PHYSICS / DYNAMICS (SCENARIO-GENERAL)
================================================================================

1. MAP / BOUNDS:
   - Square map with x,y in approximately [-bound, +bound] (default `bound=1.0`).
   - Agents spawn uniformly in [-bound, +bound] on both axes.
   - Landmarks spawn in [-(bound - 0.1), +(bound - 0.1)].

2. AGENT PHYSICS:
   - Predator radius: 0.075
   - Prey radius: 0.05
   - Predator max speed: 1.0
   - Prey max speed: 1.3 (prey are faster)
   - Action-to-force multiplier:
     - Predator: 3.0
     - Prey: 4.0

3. TAGGING / COLLISION:
   - A tag occurs when predator-prey distance is below radius sum.
   - Predator-prey collision threshold is 0.125 (= 0.075 + 0.05).
   - `respawn_at_catch` controls whether caught prey respawn (default is False).

4. DEFAULT REWARD MODE IN THIS REPO CONFIG:
   - Sparse rewards (`shape_agent_rew=False`, `shape_adversary_rew=False`).
   - Predator team gets +1.0 for capturing a stag, +0.5 for capturing a rabbit.
   - Stags require both predators simultaneously; rabbits require one predator.
   - If only one predator attempts to capture a stag, team gets -0.01.
   - Captured prey gets corresponding negative reward by type (-1.0 for stag, -0.5 for rabbit).
   - Predator rewards are shared across predators (`adversaries_share_rew=True`).
- Episode ends when predators capture one stag or two rabbits, or at 50 steps.

================================================================================
OBSERVATION SPACE (EGO PREDATOR)
================================================================================

For ego `adversary_0` in the default 2v5 setup, observation is 30D.
Sign convention for relative position is `other_pos - self_pos` (NOT the reverse).

Layout:
- [0:2]   self velocity `(vx, vy)`
- [2:4]   self position `(x, y)`
- [4:6]   landmark_0 relative position
- [6:8]   landmark_1 relative position
- [8:10]  teammate predator relative position (`adversary_1 - self`)
- [10:12] prey_0 relative position (`agent_0 - self`)
- [12:14] prey_1 relative position (`agent_1 - self`)
- [14:16] prey_2 relative position (`agent_2 - self`)
- [16:18] prey_3 relative position (`agent_3 - self`)
- [18:20] prey_4 relative position (`agent_4 - self`)
- [20:22] prey_0 absolute velocity
- [22:24] prey_1 absolute velocity
- [24:26] prey_2 absolute velocity
- [26:28] prey_3 absolute velocity
- [28:30] prey_4 absolute velocity

Notes:
- For predators, other-agent velocity terms are included for prey (not same-team predator).
- Observation dimensionality differs by team role and scales with team sizes; this index layout is for the default 2v5 ego predator setup.

================================================================================
ACTION SPACE
================================================================================

Continuous 2D force `(fx, fy)`:
- `fx` in [-1, 1]
- `fy` in [-1, 1]

There is no discrete "tag" action; tagging happens only through physical collision."""

REWARD_API_REFERENCE = """Reward function API (authoritative):
- Signature: `compute_reward(self, agent)`.
- In predator-prey training, `self` is VMAS `simple_tag` scenario and `agent` is a VMAS agent.
- Accessible state:
  - `self.world.agents`: all agents in the world.
  - `self.adversaries()`: list of predator agents.
  - `self.good_agents()`: list of prey agents.
  - `self.is_collision(a, b)`: collision test between two agents.
  - `entity.state.pos` and `entity.state.vel`: tensors with leading batch dim.
- Return value:
  - Return a torch tensor of shape `(batch_dim,)` on world device.
- Note:
  - The trainable ego agent corresponds to the first world agent (`adversary_0`)."""

REWARD_EXAMPLE = """Simple predator reward example: encourage chasing the nearest prey and add collision bonus.

```python
import torch

def compute_reward(self, agent):
    reward = torch.zeros(self.world.batch_dim, device=self.world.device)

    predators = self.adversaries()
    prey_agents = self.good_agents()

    # Distance shaping to nearest prey.
    prey_dists = torch.stack(
        [torch.linalg.vector_norm(prey.state.pos - agent.state.pos, dim=1) for prey in prey_agents],
        dim=1,
    )
    reward += -prey_dists.min(dim=1).values

    # Tag bonus.
    for prey in prey_agents:
        reward[self.is_collision(agent, prey)] += 10.0

    return reward
```"""


# ---------------------------------------------------------------------------- #
#                                 Action Type                                  #
# ---------------------------------------------------------------------------- #


class ActionType(BaseModel):
    """Continuous action for VMAS simple_tag."""

    fx: float = Field(description="Horizontal force component (-1=left, 1=right)")
    fy: float = Field(description="Vertical force component (-1=down, 1=up)")

    @classmethod
    def default_action(cls):
        return cls(fx=0.0, fy=0.0)

    def to_tensor(self):
        return torch.tensor([self.fx, self.fy], dtype=torch.float32)


# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #


def _load_task_cfg() -> dict:
    with open(RL_CFG_DIR / "simple_tag.yaml") as f:
        return yaml.safe_load(f)


def _resolve_semdiv_config(overrides: dict, task_cfg: dict) -> dict:
    """Resolve SEMDIV scenario kwargs with sane defaults."""
    cfg = dict(overrides)

    num_good_agents = int(cfg.get("num_good_agents", task_cfg.get("num_good_agents", 5)))
    num_stags = int(cfg.get("num_stags", 2))
    num_rabbits = int(cfg.get("num_rabbits", num_good_agents - num_stags))
    if num_stags < 0 or num_rabbits < 0:
        raise ValueError(
            f"num_stags and num_rabbits must be non-negative, got {num_stags=} {num_rabbits=}."
        )
    if num_stags + num_rabbits != num_good_agents:
        raise ValueError(
            f"Expected num_good_agents == num_stags + num_rabbits, got "
            f"{num_good_agents=} and {num_stags + num_rabbits=}."
        )

    return {
        "num_good_agents": num_good_agents,
        "ai_good_agents": bool(cfg.get("ai_good_agents", task_cfg.get("ai_good_agents", True))),
        "num_stags": num_stags,
        "num_rabbits": num_rabbits,
        "stag_capture_requirement": int(cfg.get("stag_capture_requirement", 2)),
        "rabbit_capture_requirement": int(cfg.get("rabbit_capture_requirement", 1)),
        "stag_capture_reward": float(cfg.get("stag_capture_reward", 1.0)),
        "rabbit_capture_reward": float(cfg.get("rabbit_capture_reward", 0.5)),
        "stag_caught_penalty": float(cfg.get("stag_caught_penalty", 1.0)),
        "rabbit_caught_penalty": float(cfg.get("rabbit_caught_penalty", 0.5)),
        "solo_stag_attempt_penalty": float(cfg.get("solo_stag_attempt_penalty", 0.01)),
        "remove_caught_prey": bool(cfg.get("remove_caught_prey", True)),
        "stags_to_end": int(cfg.get("stags_to_end", 1)),
        "rabbits_to_end": int(cfg.get("rabbits_to_end", 2)),
    }

def build_group_map(task_cfg: dict) -> dict[str, list[str]]:
    num_adversaries = int(task_cfg.get("num_adversaries", 1))
    ai_good_agents = bool(task_cfg.get("ai_good_agents", True))
    num_good_agents = int(task_cfg.get("num_good_agents", 1))
    adversaries = [f"adversary_{i}" for i in range(num_adversaries)]
    if not adversaries:
        return {}

    group_map = {"ego": [adversaries[0]]}
    if len(adversaries) > 1:
        group_map["teammate"] = adversaries[1:]
    if not ai_good_agents:
        group_map["opponents"] = [f"agent_{i}" for i in range(num_good_agents)]
    return group_map


def _set_ego_agent_render_color(env, color=EGO_AGENT_DARK_RED) -> None:
    scenario = getattr(env, "scenario", None)
    world = getattr(scenario, "world", None)
    if world is None:
        return

    for agent in world.agents:
        if getattr(agent, "name", None) == EGO_AGENT_NAME:
            agent.color = color
            return


def train_fn(reward_fn: callable, output_dir: str, **kwargs):
    """Train the ego predator policy in VMAS simple_tag with scripted non-ego agents."""
    scripted_policy = kwargs.pop("scripted_policy", None)

    task_cfg = _load_task_cfg()
    semdiv_config = _resolve_semdiv_config(overrides=kwargs, task_cfg=task_cfg)
    for key in semdiv_config:
        kwargs.pop(key, None)

    effective_task_cfg = dict(task_cfg)
    effective_task_cfg.update(semdiv_config)
    group_map = build_group_map(effective_task_cfg)

    if scripted_policy is None:
        scripted_policy = ScriptedPolicy(
            num_landmarks=int(task_cfg.get("num_landmarks", 2)),
            num_adversaries=int(task_cfg.get("num_adversaries", 2)),
            num_good_agents=int(semdiv_config["num_good_agents"]),
            num_stags=int(semdiv_config["num_stags"]),
        )
    elif isinstance(scripted_policy, torch.nn.Module):
        scripted_policy = [scripted_policy]
    else:
        scripted_policy = list(scripted_policy)

    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=simple_tag_scenario,
        scenario_name="simple_tag",
        vmas_task=VmasTask.SIMPLE_TAG,
        rl_cfg_dir=RL_CFG_DIR,
        group_map=group_map,
        trainable_group=default_trainable_group(group_map),
        teammate_policy=scripted_policy,
        scenario_kwargs=semdiv_config,
        **kwargs,
    )


def rollout_fn(policy, env=None, **kwargs):
    """Rollout the policy in VMAS simple_tag."""
    if env is None:
        env = create_env(**kwargs)
    _set_ego_agent_render_color(env)

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 50

    return rollout_vmas_env(policy=policy, env=env, **kwargs)


def serialize_state_fn(rollout_data: list[dict]) -> list[dict]:
    num_landmarks, _, num_good_agents, num_stags = _resolve_scripted_counts()
    collision_threshold = 0.125
    min_expected_len = 4 + 2 * num_landmarks + 2 + 4 * num_good_agents
    for step in rollout_data:
        for team_name in ("ego_team", "opponent_team"):
            for agent_state in step.get(team_name, {}).values():
                raw_state = agent_state.get("state")
                if isinstance(raw_state, list) and len(raw_state) >= min_expected_len:
                    self_pos = [float(raw_state[2]), float(raw_state[3])]
                    self_vel = [float(raw_state[0]), float(raw_state[1])]

                    landmarks = []
                    offset = 4
                    for i in range(num_landmarks):
                        rel = [float(raw_state[offset + 2 * i]), float(raw_state[offset + 2 * i + 1])]
                        landmarks.append(
                            {
                                "slot": i,
                                "relative_pos": rel,
                                "pos": [self_pos[0] + rel[0], self_pos[1] + rel[1]],
                            }
                        )

                    teammate_offset = offset + 2 * num_landmarks
                    teammate_rel = [float(raw_state[teammate_offset]), float(raw_state[teammate_offset + 1])]
                    prey_offset = teammate_offset + 2
                    prey_vel_offset = prey_offset + 2 * num_good_agents
                    prey = []
                    nearest_prey_slot = None
                    nearest_prey_distance = None
                    nearest_stag_slot = None
                    nearest_stag_distance = None
                    nearest_rabbit_slot = None
                    nearest_rabbit_distance = None
                    teammate_pos = [self_pos[0] + teammate_rel[0], self_pos[1] + teammate_rel[1]]
                    teammate_distance = float((teammate_rel[0] ** 2 + teammate_rel[1] ** 2) ** 0.5)

                    for i in range(num_good_agents):
                        rel = [float(raw_state[prey_offset + 2 * i]), float(raw_state[prey_offset + 2 * i + 1])]
                        vel = [float(raw_state[prey_vel_offset + 2 * i]), float(raw_state[prey_vel_offset + 2 * i + 1])]
                        dist = float((rel[0] ** 2 + rel[1] ** 2) ** 0.5)
                        teammate_rel_to_prey = [rel[0] - teammate_rel[0], rel[1] - teammate_rel[1]]
                        teammate_dist = float((teammate_rel_to_prey[0] ** 2 + teammate_rel_to_prey[1] ** 2) ** 0.5)
                        prey_type = "stag" if i < num_stags else "rabbit"
                        capture_requirement = 2 if prey_type == "stag" else 1
                        ego_in_tag_range = dist <= collision_threshold
                        teammate_in_tag_range = teammate_dist <= collision_threshold
                        coordinated_tag_ready = capture_requirement == 1 and ego_in_tag_range
                        if capture_requirement == 2:
                            coordinated_tag_ready = ego_in_tag_range and teammate_in_tag_range
                        if nearest_prey_distance is None or dist < nearest_prey_distance:
                            nearest_prey_distance = dist
                            nearest_prey_slot = i
                        if prey_type == "stag" and (nearest_stag_distance is None or dist < nearest_stag_distance):
                            nearest_stag_distance = dist
                            nearest_stag_slot = i
                        if prey_type == "rabbit" and (nearest_rabbit_distance is None or dist < nearest_rabbit_distance):
                            nearest_rabbit_distance = dist
                            nearest_rabbit_slot = i
                        prey.append(
                            {
                                "slot": i,
                                "type": prey_type,
                                "relative_pos": rel,
                                "pos": [self_pos[0] + rel[0], self_pos[1] + rel[1]],
                                "velocity": vel,
                                "distance": dist,
                                "teammate_distance": teammate_dist,
                                "capture_requirement": capture_requirement,
                                "ego_in_tag_range": ego_in_tag_range,
                                "teammate_in_tag_range": teammate_in_tag_range,
                                "coordinated_tag_ready": coordinated_tag_ready,
                                "solo_stag_attempt": prey_type == "stag" and ego_in_tag_range and not teammate_in_tag_range,
                            }
                        )

                    agent_state["serialized_state"] = {
                        "self_pos": self_pos,
                        "self_vel": self_vel,
                        "teammate_pos": teammate_pos,
                        "teammate_distance": teammate_distance,
                        "landmarks": landmarks,
                        "prey": prey,
                        "nearest_prey_slot": nearest_prey_slot,
                        "nearest_prey_distance": nearest_prey_distance,
                        "nearest_stag_slot": nearest_stag_slot,
                        "nearest_stag_distance": nearest_stag_distance,
                        "nearest_rabbit_slot": nearest_rabbit_slot,
                        "nearest_rabbit_distance": nearest_rabbit_distance,
                        "coordinated_stag_ready": any(item["type"] == "stag" and item["coordinated_tag_ready"] for item in prey),
                        "rabbit_tag_ready": any(item["type"] == "rabbit" and item["coordinated_tag_ready"] for item in prey),
                        "solo_stag_attempt": any(item["solo_stag_attempt"] for item in prey),
                    }
    return rollout_data


# ---------------------------------------------------------------------------- #
#                              Make env function                               #
# ---------------------------------------------------------------------------- #


def create_env(num_envs=1, continuous_actions=True, seed=None, device="cpu", **kwargs):
    """Create VMAS simple_tag environment."""
    task_cfg = _load_task_cfg()
    semdiv_config = _resolve_semdiv_config(overrides=kwargs, task_cfg=task_cfg)
    num_good_agents = int(semdiv_config["num_good_agents"])

    task = VmasTask.SIMPLE_TAG.get_from_yaml(path=str(RL_CFG_DIR / "simple_tag.yaml"))
    config = dict(task.config)
    config["num_good_agents"] = num_good_agents
    config.update(semdiv_config)
    config["group_map"] = build_group_map(config)

    env = VmasEnv(
        scenario=simple_tag_scenario.Scenario(),
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )
    ego_group = list(getattr(env, "group_map", {}).get("ego", []))
    if len(ego_group) != 1 or ego_group[0] != EGO_AGENT_NAME:
        raise ValueError(
            f"Invalid simple_tag ego mapping: expected ['{EGO_AGENT_NAME}'], got {ego_group}."
        )
    _set_ego_agent_render_color(env)
    return env


# ---------------------------------------------------------------------------- #
#                              Scripted Policy                                 #
# ---------------------------------------------------------------------------- #


def _safe_normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True).clamp_min(eps)
    return vec / norm


class ScriptedPolicy(torch.nn.Module):
    """Scripted predator partner policy for simple_tag."""

    def __init__(self, env=None, num_landmarks=None, num_adversaries=None, num_good_agents=None, num_stags=None):
        super().__init__()
        num_landmarks, num_adversaries, num_good_agents, num_stags = _resolve_scripted_counts(
            env=env,
            num_landmarks=num_landmarks,
            num_adversaries=num_adversaries,
            num_good_agents=num_good_agents,
            num_stags=num_stags,
        )
        self.num_landmarks = num_landmarks
        self.num_adversaries = num_adversaries
        self.num_good_agents = num_good_agents
        self.num_stags = max(0, min(num_stags, self.num_good_agents))
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(f"PP ScriptedPolicy expected torch.Tensor observation, got {type(obs_tensor).__name__}.")
        if obs_tensor.dim() != 2:
            raise ValueError(
                f"PP ScriptedPolicy expects observation tensor with shape (batch, obs_dim), got {tuple(obs_tensor.shape)}."
            )

        batch_size = obs_tensor.shape[0]
        obs_dim = obs_tensor.shape[-1]

        start = 4 + 2 * self.num_landmarks
        n_other_pos = (self.num_adversaries - 1) + self.num_good_agents
        end = start + 2 * n_other_pos

        action = torch.zeros(batch_size, 2, device=obs_tensor.device, dtype=torch.float32)
        if obs_dim < end or n_other_pos <= 0 or self.num_good_agents <= 0:
            return action

        landmark_flat = obs_tensor[:, 4:start]
        landmarks = (
            landmark_flat.reshape(batch_size, self.num_landmarks, 2)
            if landmark_flat.shape[-1] >= 2
            else None
        )
        other_pos = obs_tensor[:, start:end].reshape(batch_size, n_other_pos, 2)

        teammate_count = max(0, self.num_adversaries - 1)
        teammate_pos = other_pos[:, :teammate_count] if teammate_count > 0 else None
        prey_pos = other_pos[:, teammate_count:]

        prey_dist = torch.linalg.vector_norm(prey_pos, dim=-1)
        score = prey_dist.clone()

        if teammate_pos is not None and self.num_stags > 0:
            tm = teammate_pos[:, 0, :].unsqueeze(1)
            stag_pos = prey_pos[:, : self.num_stags, :]
            stag_dist = prey_dist[:, : self.num_stags]
            teammate_to_stag_dist = torch.linalg.vector_norm(stag_pos - tm, dim=-1)
            lone_contact = (stag_dist < 0.18) & (teammate_to_stag_dist > 0.25)
            score[:, : self.num_stags] += 0.35 * lone_contact.to(score.dtype)
            score[:, : self.num_stags] += 0.12 * torch.relu(teammate_to_stag_dist - 0.2)

        target_idx = score.argmin(dim=-1)
        target = prey_pos[torch.arange(batch_size, device=obs_tensor.device), target_idx]
        action = _safe_normalize(target)

        if teammate_pos is not None:
            # Keep mild spatial separation from teammate to avoid path collapse.
            sep = -_safe_normalize(teammate_pos[:, 0, :])
            action = action + 0.2 * sep

        if landmarks is not None:
            # Patrol landmarks when no nearby prey is visible.
            min_prey_dist = prey_dist.min(dim=-1).values
            lm_idx = torch.linalg.vector_norm(landmarks, dim=-1).argmin(dim=-1)
            lm_target = landmarks[torch.arange(batch_size, device=obs_tensor.device), lm_idx]
            lm_action = _safe_normalize(lm_target)
            far_mask = (min_prey_dist > 0.8).unsqueeze(-1)
            action = torch.where(far_mask, 0.6 * action + 0.4 * lm_action, action)

        action = _safe_normalize(action).clamp(-1.0, 1.0)
        return action


def _resolve_scripted_counts(
    env=None,
    num_landmarks=None,
    num_adversaries=None,
    num_good_agents=None,
    num_stags=None,
) -> tuple[int, int, int, int]:
    task_cfg = _load_task_cfg()

    if env is not None:
        scenario = getattr(env, "scenario", None)
        if num_landmarks is None:
            world = getattr(scenario, "world", None)
            landmarks = getattr(world, "landmarks", None)
            if landmarks is not None:
                num_landmarks = int(len(landmarks))
        if num_adversaries is None and scenario is not None and hasattr(scenario, "adversaries"):
            num_adversaries = int(len(scenario.adversaries()))
        if num_good_agents is None and scenario is not None and hasattr(scenario, "good_agents"):
            num_good_agents = int(len(scenario.good_agents()))
        if num_stags is None and scenario is not None and hasattr(scenario, "num_stags"):
            num_stags = int(scenario.num_stags)

    if num_landmarks is None:
        num_landmarks = int(task_cfg.get("num_landmarks", 2))
    if num_adversaries is None:
        num_adversaries = int(task_cfg.get("num_adversaries", 2))
    if num_good_agents is None:
        num_good_agents = int(task_cfg.get("num_good_agents", 5))
    if num_stags is None:
        num_stags = 2

    return int(num_landmarks), int(num_adversaries), int(num_good_agents), int(num_stags)
