from pathlib import Path

import torch
import yaml
from benchmarl.environments import VmasTask
from pydantic import BaseModel, Field
from vmas.scenarios import football as football_scenario

from popgen.envs.vmas_common import default_trainable_group, rollout_vmas_env, train_vmas_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

ENVIRONMENT_DESCRIPTION = """The environment is a football (soccer) game in simulation.
The objective of the blue team is to score by getting the ball into the red team's goal area while preventing the red team from scoring in the blue team's goal area.
The blue team has 2 agents and the red team has 1 agent (2v1), and blue agents must coordinate without explicit communication.
An episode terminates immediately when blue scores or when blue concedes.

There are two blue agents: Agent 0 and Agent 1.
The ego team is on the left side wearing blue jerseys, the opponent team is on the right side wearing red jerseys.
The ego agent is the blue football player tagged #1 (blue Agent 0).

================================================================================
CRITICAL PHYSICS PARAMETERS - READ CAREFULLY
================================================================================

1. FIELD DIMENSIONS:
   - Pitch size: 3.0 (x) x 1.5 (y)
   - X-axis: -1.5 (left) to +1.5 (right)
   - Y-axis: -0.75 (bottom) to +0.75 (top)
   - Field center: (0.0, 0.0)

2. AGENT PHYSICS:
   - Agent radius: 0.025
   - Maximum speed: 0.15 units/timestep
   - Action magnitude 1.0 corresponds to acceleration of 0.1 units/timestep^2

3. BALL PHYSICS:
   - Ball radius: 0.02
   - Ball has friction/drag and slows naturally

4. BALL CONTACT (VERY IMPORTANT):
   - Contact threshold distance: 0.045 (= 0.025 + 0.02)
   - This contact zone is very small relative to field size, so accurate positioning is required to move the ball

5. GOAL POSITIONS:
   - Blue goal to defend: (-1.5, 0.0) on LEFT
   - Red goal to attack: (+1.51, 0.0) on RIGHT
   - To score, move/push the ball toward +X into the right goal area

================================================================================
OBSERVATION TENSOR LAYOUT (32 dimensions in this 2v1 config)
================================================================================

Each agent observes a flattened 32-dimensional vector.
Sign conventions matter: relative terms are written in exact order (e.g. `agent_pos - ball_pos`, not the reverse).

  [0:2]   - `agent_force` (2D): current force applied to ego agent
  [2:4]   - `agent_pos - ball_pos` (2D): relative position from ball to ego
  [4:6]   - `agent_vel - ball_vel` (2D): relative velocity between ego and ball
  [6:8]   - `ball_pos - red_goal_pos` (2D): ball displacement relative to scoring target
  [8:10]  - `ball_vel` (2D): absolute ball velocity
  [10:12] - `ball_force` (2D): current force on ball
  [12:14] - `agent_pos - red_goal_pos` (2D): ego position relative to scoring target
  [14:16] - `agent_vel` (2D): absolute ego velocity
  [16:24] - adversaries (8D): 1 adversary x 8 dims
             For each adversary: relative position, relative velocity, absolute velocity, force
  [24:32] - teammate (8D): 1 teammate x 8 dims
             Relative position, relative velocity, teammate absolute velocity, teammate force

Adversary features help anticipate opponent pressure and interceptions.
Teammate features help spacing and coordination.

================================================================================
ACTION OUTPUT
================================================================================

The action should be a torch tensor of shape `(1, 2)` or `(2,)` representing force `(fx, fy)`.
- `fx` in [-1, 1]: left/right force
- `fy` in [-1, 1]: down/up force
- Action magnitude must be <= 1.0 (larger values can fail rollout)

CRITICAL NOTE:
- `enable_shooting` is False.
- There is no separate kick/shoot action.
- Ball movement is only through physical collisions, so dribbling and body positioning are key."""

REWARD_API_REFERENCE = """Reward function API (authoritative):
- Signature: `compute_reward(self, agent)`.
- In football training, `self` is the VMAS football scenario (`Scenario`) and `agent` is one VMAS agent.
- Accessible state:
  - `self.world`: world object.
  - `self.world.batch_dim`: number of parallel environments.
  - `self.world.blue_agents`, `self.world.red_agents`: team agent lists.
  - `self.world.ball`: ball entity.
  - For any entity `e`: `e.state.pos` and `e.state.vel` are tensors of shape `(batch_dim, 2)`.
  - Typical scenario attributes used in default reward: `self.pitch_length`, `self.goal_size`, `self.ball_size`, `self.scoring_reward`.
- Return value:
  - Return a torch tensor of shape `(batch_dim,)` on the same device as world tensors."""

REWARD_EXAMPLE = """Simple reward function example for the football environment:

```python
import torch

def compute_reward(self, agent):

    reward = torch.zeros(self.world.batch_dim, device=self.world.device)

    # You are designing reward for 'agent' in 'world'
    # You can access to list of blue team agents and red team agents via self.world.blue_agents and self.world.red_agents
    # You can also access the ball state via self.world.ball.

    blue_team = self.world.blue_agents # List of blue team agents
    red_team = self.world.red_agents # List of red team agents
    ball = self.world.ball

    # You can access to agent or ball's state via agent.state
    # agent.state.pos gives you the position of the agent in torch tensor shape (batch_dim, 2)
    # agent.state.vel gives you the velocity of the agent in torch tensor shape (batch_dim, 2)
    # ball.state.pos gives you the position of the ball in torch tensor shape (batch_dim, 2)
    # ball.state.vel gives you the velocity of the ball in torch tensor shape (batch_dim

    # This computes the sparse reward for scoring or conceding a goal. You can design your own reward function based on the game state.
    # If the ball crosses the right goal line within the goal area, the blue team scores and gets a positive reward.
    # If the ball crosses the left goal line within the goal area, the red team scores and the blue team gets a negative reward.
    over_right_line = (
        self.ball.state.pos[:, X] > self.pitch_length / 2 + self.ball_size / 2
    )
    over_left_line = (
        self.ball.state.pos[:, X] < -self.pitch_length / 2 - self.ball_size / 2
    )
    goal_mask = (self.ball.state.pos[:, Y] <= self.goal_size / 2) * (
        self.ball.state.pos[:, Y] >= -self.goal_size / 2
    )
    blue_score = over_right_line * goal_mask
    red_score = over_left_line * goal_mask
    self._sparse_reward_blue = (
        self.scoring_reward * blue_score - self.scoring_reward * red_score
    )

    self._done = blue_score | red_score

    reward = self._sparse_reward_blue

    return reward
```"""


def _set_done_on_goal_or_conceded(scenario) -> None:
    """Force terminal state when either team scores."""
    world = getattr(scenario, "world", None)
    ball = getattr(world, "ball", None) if world is not None else None
    if ball is None or not hasattr(ball, "state"):
        return
    ball_pos = getattr(ball.state, "pos", None)
    if ball_pos is None:
        return

    over_right_line = ball_pos[:, 0] > scenario.pitch_length / 2 + scenario.ball_size / 2
    over_left_line = ball_pos[:, 0] < -scenario.pitch_length / 2 - scenario.ball_size / 2
    goal_mask = (ball_pos[:, 1] <= scenario.goal_size / 2) & (ball_pos[:, 1] >= -scenario.goal_size / 2)
    blue_score = over_right_line & goal_mask
    red_score = over_left_line & goal_mask
    scenario._done = blue_score | red_score


def _wrap_reward_with_goal_termination(reward_fn: callable):
    def wrapped_reward_fn(scenario, agent):
        reward = reward_fn(scenario, agent)
        _set_done_on_goal_or_conceded(scenario)
        return reward

    return wrapped_reward_fn

# ---------------------------------------------------------------------------- #
#                                 Action Type                                  #
# ---------------------------------------------------------------------------- #


class ActionType(BaseModel):
    """
    Action type for the football environment.
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


def build_group_map(task_cfg: dict) -> dict[str, list[str]]:
    raw_group_map = task_cfg.get("group_map")
    if not raw_group_map:
        # Derive from n_blue_agents / n_red_agents when group_map isn't explicitly set.
        n_blue = task_cfg.get("n_blue_agents", 0)
        if not n_blue:
            raise ValueError("Football task config must define either 'group_map' or 'n_blue_agents'.")
        raw_group_map = {"agent_blue": [f"agent_blue_{i}" for i in range(n_blue)]}
        if not task_cfg.get("ai_red_agents", True):
            n_red = task_cfg.get("n_red_agents", 0)
            if n_red:
                raw_group_map["agent_red"] = [f"agent_red_{i}" for i in range(n_red)]
    if "agent_blue" not in raw_group_map:
        raise ValueError(f"Football task config must define 'agent_blue' in group_map, got {raw_group_map}.")

    blue_agents = list(raw_group_map["agent_blue"])
    if not blue_agents:
        raise ValueError("Football group_map must contain at least one blue agent.")

    group_map = {"ego": [blue_agents[0]]}
    if len(blue_agents) > 1:
        group_map["teammate"] = blue_agents[1:]

    red_agents = list(raw_group_map.get("agent_red", []))
    if red_agents:
        group_map["opponents"] = red_agents

    return group_map


def train_fn(reward_fn: callable, output_dir: str, **kwargs):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward tensor.
        output_dir: Directory to save results.
    """
    reward_fn = _wrap_reward_with_goal_termination(reward_fn)
    scripted_policy = kwargs.pop("scripted_policy", None)
    if scripted_policy is None:
        temp_env = create_env(num_envs=1, seed=0, device="cpu")
        try:
            scripted_policy = ScriptedPolicy(temp_env)
        finally:
            temp_env.close()
        # env info is captured in __init__; clear the reference so the module
        # remains picklable when BenchMARL sends it to data-collector workers.
        scripted_policy.env = None

    task = VmasTask.FOOTBALL.get_from_yaml(path=str(RL_CFG_DIR / "football.yaml"))
    group_map = build_group_map(task.config)

    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=football_scenario,
        scenario_name="football",
        vmas_task=VmasTask.FOOTBALL,
        rl_cfg_dir=RL_CFG_DIR,
        group_map=group_map,
        trainable_group=default_trainable_group(group_map),
        teammate_policy=[scripted_policy],
        **kwargs,
    )


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy loaded from the checkpoint in save_dir.
    Uses rollout states extracted directly from tensordicts.
    """
    # if env is not defined, load default football environment
    if env is None:
        env = create_env(**kwargs)

    if "max_steps" not in kwargs:
        # Match football task horizon from rl_cfg/football.yaml unless explicitly overridden.
        kwargs["max_steps"] = 300

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
                if isinstance(raw_state, list) and len(raw_state) >= 32:
                    ego_x = 1.51 + float(raw_state[12])
                    ego_y = float(raw_state[13])
                    ball_x = 1.51 + float(raw_state[6])
                    ball_y = float(raw_state[7])
                    teammate_x = ego_x + float(raw_state[24])
                    teammate_y = ego_y + float(raw_state[25])
                    opponent_x = ego_x + float(raw_state[16])
                    opponent_y = ego_y + float(raw_state[17])
                    ego_ball_dx = float(raw_state[2])
                    ego_ball_dy = float(raw_state[3])
                    agent_state["serialized_state"] = {
                        "ego_pos": [ego_x, ego_y],
                        "ego_speed": float(torch.linalg.vector_norm(torch.tensor(raw_state[14:16], dtype=torch.float32)).item()),
                        "ball_pos": [ball_x, ball_y],
                        "teammate_pos": [teammate_x, teammate_y],
                        "opponent_pos": [opponent_x, opponent_y],
                        "red_goal_pos": [1.51, 0.0],
                        "blue_goal_pos": [-1.5, 0.0],
                        "distance_to_ball": float((ego_ball_dx**2 + ego_ball_dy**2) ** 0.5),
                        "distance_to_teammate": float(((teammate_x - ego_x) ** 2 + (teammate_y - ego_y) ** 2) ** 0.5),
                        "in_opponent_half": ego_x > 0.0,
                        "in_final_third": ego_x > 0.5,
                        "in_box": ego_x > 1.0,
                        "central_lane": abs(ego_y) <= 0.2,
                        "ball_in_final_third": ball_x > 0.5,
                        "ball_in_box": ball_x > 1.0,
                    }
    return rollout_data

# ---------------------------------------------------------------------------- #
#                              Make env function                               #
# ---------------------------------------------------------------------------- #

def create_env(num_envs=1, continuous_actions=True, seed=None, device="cpu", **kwargs):
    """
    Creates the football environment using VMAS.

    Args:
        kwargs: Keyword arguments to pass to the environment creation function. Can include:
            num_envs, device, continuous_actions, max_steps, seed, dict_spaces, multidiscrete_actions, clamp_actions, grad_enabled, terminated_truncated, wrapper_kwargs, etc.

    Returns:
        An instance of the football environment.
    """
    task = VmasTask.FOOTBALL.get_from_yaml(path=str(RL_CFG_DIR / "football.yaml"))
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


class ScriptedPolicy(torch.nn.Module):
    """Hybrid scripted football partner policy.

    Behavior:
    - If teammate is closer to the ball, yield possession and run toward goal.
    - Otherwise recover the ball and, once in control, move to pass toward teammate.
    """

    def __init__(self, env=None):
        super().__init__()
        self.env = env
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.pitch_length = 3.0
        self.teammate_ball_radius = 0.16
        self.teammate_arrival_radius = 0.20
        self.teammate_arrival_horizon = 3.0
        self.ball_control_radius = 0.08
        self.teammate_support_radius = 0.35

    @staticmethod
    def _unit(direction: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8
        return direction / norm

    @staticmethod
    def _teammate_rel_slice(obs_dim: int) -> slice:
        # Observation layout:
        # 16 base dims + (8 * n_adversaries) + 8 teammate dims
        # teammate relative position is the first 2 dims of the teammate block.
        n_adversaries = max(0, (obs_dim - 24) // 8)
        start = 16 + 8 * n_adversaries
        return slice(start, start + 2)

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(f"Football ScriptedPolicy expected torch.Tensor observation, got {type(obs_tensor).__name__}.")
        if obs_tensor.dim() != 2:
            raise ValueError(
                f"Football ScriptedPolicy expects observation tensor with shape (batch, obs_dim), got {tuple(obs_tensor.shape)}."
            )

        if obs_tensor.shape[-1] < 16:
            # Fallback for unexpected observation schemas.
            direction_to_ball = -obs_tensor[:, 2:4]
            action = self._unit(direction_to_ball).clamp(-1, 1)
            return action

        # Core vectors from observation.
        rel_ball = obs_tensor[:, 2:4]           # agent_pos - ball_pos
        rel_goal = obs_tensor[:, 12:14]         # agent_pos - opponent_goal_pos
        dir_to_ball = self._unit(-rel_ball)
        dir_to_goal = self._unit(-rel_goal)

        teammate_slice = self._teammate_rel_slice(obs_tensor.shape[-1])
        teammate_start = teammate_slice.start
        rel_teammate = obs_tensor[:, teammate_slice]  # agent_pos - teammate_pos
        dir_to_teammate = self._unit(-rel_teammate)
        teammate_vel = obs_tensor[:, teammate_start + 4 : teammate_start + 6]
        ball_vel = obs_tensor[:, 8:10]

        dist_self_ball = torch.linalg.norm(rel_ball, dim=-1, keepdim=True)
        teammate_to_ball = rel_ball - rel_teammate
        dist_tm_ball = torch.linalg.norm(teammate_to_ball, dim=-1, keepdim=True)
        teammate_to_ball_vel = teammate_vel - ball_vel
        teammate_to_ball_pred = teammate_to_ball + self.teammate_arrival_horizon * teammate_to_ball_vel
        dist_tm_ball_pred = torch.linalg.norm(teammate_to_ball_pred, dim=-1, keepdim=True)

        teammate_closer = dist_tm_ball + 0.02 < dist_self_ball
        teammate_near_ball = dist_tm_ball < self.teammate_ball_radius
        teammate_arriving_ball = (dist_tm_ball_pred + 0.01 < dist_tm_ball) & (
            dist_tm_ball_pred < self.teammate_arrival_radius
        )
        teammate_ready_ball = teammate_near_ball | teammate_arriving_ball
        yield_to_teammate = teammate_closer & teammate_ready_ball
        has_ball = dist_self_ball < self.ball_control_radius

        # If teammate can take possession, move into a pass-support lane:
        # closer to teammate with mild goal bias, but avoid entering the goal mouth.
        support_action = self._unit(0.65 * dir_to_teammate + 0.35 * dir_to_goal)
        away_from_goal = self._unit(rel_goal)
        too_close_to_goal = rel_goal[..., 0:1] > -0.35
        near_goal_mouth = torch.abs(rel_goal[..., 1:2]) < 0.25
        support_safe = torch.where(
            too_close_to_goal & near_goal_mouth,
            self._unit(0.85 * away_from_goal + 0.15 * dir_to_teammate),
            support_action,
        )

        # Ball progression policy by x-field zone:
        # 0-50%: carry forward; 50-75%: pass if teammate around/in front, else shoot; 75-100%: shoot.
        dist_to_goal_x = -rel_goal[:, 0:1]
        progress_x = ((self.pitch_length - dist_to_goal_x) / self.pitch_length).clamp(0.0, 1.0)
        in_build_up_zone = progress_x < 0.50
        in_pass_zone = (progress_x >= 0.50) & (progress_x <= 0.75)

        teammate_dist = torch.linalg.norm(rel_teammate, dim=-1, keepdim=True)
        teammate_around = teammate_dist < self.teammate_support_radius
        teammate_in_front = rel_teammate[:, 0:1] < -0.02
        teammate_pass_option = teammate_around | teammate_in_front

        dribble_action = self._unit(0.9 * dir_to_goal + 0.1 * dir_to_teammate)
        pass_action = self._unit(0.9 * dir_to_teammate + 0.1 * dir_to_goal)
        shoot_action = self._unit(0.96 * dir_to_goal + 0.04 * dir_to_teammate)
        pass_or_shoot = torch.where(teammate_pass_option, pass_action, shoot_action)
        attack_action = torch.where(
            in_build_up_zone,
            dribble_action,
            torch.where(in_pass_zone, pass_or_shoot, shoot_action),
        )

        # If teammate is not near the ball, prioritize ball recovery.
        recover_or_attack = torch.where(has_ball, attack_action, dir_to_ball)
        action = torch.where(yield_to_teammate, support_safe, recover_or_attack)
        action = action.clamp(-1, 1)
        return action
