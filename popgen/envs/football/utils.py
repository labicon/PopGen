from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch
import vmas
from benchmarl.environments import VmasTask
from pydantic import BaseModel, Field
from vmas.scenarios import football as football_scenario

from popgen.envs.football import football_scenario as local_football_scenario
from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

ENVIRONMENT_DESCRIPTION = """The environment is a football (soccer) game in simulation.
The blue team has 2 agents and the red team has 1 agent (2v1), and blue agents must coordinate without explicit communication.
An episode terminates immediately when blue scores or when blue concedes.

The blue team has TWO ways to earn reward, and they cannot both be done by the same agent:
  1. ATTACKING: advance the ball toward the red goal at +X, and score by putting it in.
  2. NEUTRALISING: mark the red agent by staying within 0.15 of it.  A marked red agent
     is frozen (see OPPONENT BEHAVIOUR below) and the blue team is paid a small reward
     every timestep it stays frozen.
Because there is one ball and one opponent, the highest-scoring joint behaviour is for the
two blue agents to take DIFFERENT roles: one attacks while the other marks.  Two attackers
leave the red agent free to contest the ball; two markers never advance the ball at all.
Which of the two agents takes which role is not fixed - either assignment works equally
well, so an agent has to infer from its teammate's behaviour which role is left for it.

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
OPPONENT BEHAVIOUR - THE STANDOFF RULE (IMPORTANT)
================================================================================

The red agent is a scripted AI that chases and dribbles the ball, EXCEPT:
  - While any blue agent is within 0.15 of the red agent, the red agent applies
    ZERO force and coasts to a halt.  It is neutralised for as long as it is marked.
  - Marking costs nothing but the marker's own time: it does not need ball contact.

Consequences you should exploit:
  - A blue agent that parks within 0.15 of the red agent removes the only defender,
    leaving its teammate an uncontested path to goal.
  - Marking pays a small dense reward every timestep, so a marker is productive even
    while the ball is nowhere near it.
  - The marking distance (0.15) is much larger and far easier to hold than the ball
    contact threshold (0.045), so marking is a far more reliable behaviour than dribbling.

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
  - `self.neutralize_radius` (0.15): a red agent is frozen while a blue agent is within
    this distance of it, so `dist(blue, red) < self.neutralize_radius` is the test for
    "this opponent is currently neutralised".
- Return value:
  - Return a torch tensor of shape `(batch_dim,)` on the same device as world tensors.
  - Keep the per-step reward within [-10, 10]. Values outside that range, and any NaN or inf, are clamped at runtime.
  - Keep dense per-step terms small, around |r| <= 1, and reserve larger values for rare events (order 1-5). The clamp bounds a single step, not the episode: a dense term of 1 per step already accumulates to hundreds over an episode, and returns that large make advantages explode, which drives the policy to NaN and kills the run. Aim for an episode return in the low tens.
  - Prefer normalized or tanh-shaped dense terms over raw distances or unbounded sums."""

REWARD_EXAMPLE = """API reference only -- the tensor shapes and state access are the point, not what it rewards. It rewards generic task success; your reward must instead single out the given behavior description.

Simple reward function example for the football environment:

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


def train_fn(
    reward_fn: Callable | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    **kwargs,
):
    """Trains agents using BenchMARL against a football teammate.

    Args:
        reward_fn: Optional custom reward callable (scenario, agent) → tensor.
                   If None, the environment's built-in football reward is used.
        output_dir: Directory to save results.
        run_name: Name for the training run.
        **kwargs: Forwarded to train_vmas_env, including:
            teammate_policy: list[nn.Module] or list[list[nn.Module]].
                             Flat list → rl_reward config.
                             List of lists → population_play config.
                             None → default ScriptedPolicy teammate.
    """
    if reward_fn is not None:
        reward_fn = _wrap_reward_with_goal_termination(reward_fn)

    teammate_policy = kwargs.pop("teammate_policy", None)
    if teammate_policy is None:
        teammate_policy = [ScriptedPolicy()]

    task = VmasTask.FOOTBALL.get_from_yaml(path=str(RL_CFG_DIR / "football.yaml"))
    group_map = build_group_map(task.config)

    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=local_football_scenario,
        scenario_name="football",
        vmas_task=VmasTask.FOOTBALL,
        rl_cfg_dir=RL_CFG_DIR,
        group_map=group_map,
        teammate_policy=teammate_policy,
        run_name=run_name,
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

@contextmanager
def _use_local_scenario():
    """Make `vmas.make_env("football", ...)` build the popgen scenario subclass.

    BenchMARL resolves the scenario by name, which yields the stock VMAS football
    scenario -- so without this, rollout/evaluation envs silently ignore the popgen
    scenario kwargs (`red_standoff_radius`, `neutralize_reward`, the ai_*_strength
    ranges) and run different dynamics from training, which applies the same patch
    inside `train_vmas_env`.
    """
    original_make_env = vmas.make_env

    def patched_make_env(scenario, *args, **kwargs):
        if scenario == "football":
            return original_make_env(local_football_scenario.Scenario(), *args, **kwargs)
        return original_make_env(scenario, *args, **kwargs)

    vmas.make_env = patched_make_env
    try:
        yield
    finally:
        vmas.make_env = original_make_env


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
    with _use_local_scenario():
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


# ---------------------------------------------------------------------------- #
#                          Evaluation Scripted Policies                        #
# ---------------------------------------------------------------------------- #


def _unit(direction: torch.Tensor) -> torch.Tensor:
    return (direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8)).clamp(-1, 1)


class DirectAttackerPolicy(torch.nn.Module):
    """Push the ball toward the opponent's goal by approaching from behind it.

    The action always targets either the ball or a point behind it (never the
    goal itself), so it never runs away from the ball to chase the goal. It
    continuously blends between two targets based on how well an approach to
    the ball is already aligned with driving it toward goal:
    - Well aligned (agent is already goal-side of... i.e. behind... the ball
      relative to goal): drive straight at the ball, which pushes it forward.
    - Poorly aligned (e.g. agent is between the ball and the goal): circle to
      a point on the far side of the ball from goal before engaging.
    Blending continuously on alignment (instead of switching on a fixed
    distance radius) avoids bang-bang oscillation near a threshold boundary.

    Ignores teammates — useful as a selfish attacker behavior baseline.
    """

    approach_offset: float = 0.12  # how far behind the ball to circle to
    align_low: float = 0.2   # alignment below which the policy fully circles around
    align_high: float = 0.8  # alignment above which the policy fully drives at the ball

    def __init__(self, env=None):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rel_ball = obs[:, 2:4]    # agent_pos - ball_pos
        rel_goal = obs[:, 12:14]  # agent_pos - opponent_goal_pos

        dir_to_ball = _unit(-rel_ball)                 # agent -> ball
        dir_ball_to_goal = _unit(rel_ball - rel_goal)   # ball -> goal

        # Point on the far side of the ball from goal; heading there first
        # lines the agent up to push the ball forward instead of past it.
        approach_point_rel = rel_ball + self.approach_offset * dir_ball_to_goal
        dir_to_approach = _unit(-approach_point_rel)

        # How well "toward the ball" already agrees with "toward goal".
        alignment = (dir_to_ball * dir_ball_to_goal).sum(dim=-1, keepdim=True)
        weight = ((alignment - self.align_low) / (self.align_high - self.align_low)).clamp(0.0, 1.0)

        action = weight * dir_to_ball + (1.0 - weight) * dir_to_approach
        return _unit(action)


class WideSupportPolicy(torch.nn.Module):
    """Provide a wide passing lane rather than directly chasing the ball.

    Drifts to a position wide of the teammate (y offset) and slightly ahead
    in the attacking direction. Falls back to direct ball-chase when very far.
    """

    wide_offset: float = 0.35   # lateral distance from teammate
    fallback_dist: float = 1.2  # switch to direct chase when far from ball

    def __init__(self, env=None):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rel_ball = obs[:, 2:4]    # agent_pos - ball_pos → negate → toward ball
        rel_goal = obs[:, 12:14]  # agent_pos - goal_pos → negate → toward goal

        # Derive the teammate block rather than hardcoding slice(24, 26): the offset
        # shifts with n_red_agents.
        tm_slice = ScriptedPolicy._teammate_rel_slice(obs.shape[-1])
        rel_teammate = obs[:, tm_slice]  # agent_pos - teammate_pos

        dist_to_ball = torch.linalg.norm(rel_ball, dim=-1, keepdim=True)
        dir_to_ball = _unit(-rel_ball)
        dir_to_goal = _unit(-rel_goal)

        # Target: slightly ahead of teammate (in +x) and displaced laterally.
        # rel_teammate points from teammate toward ego; negate → toward teammate.
        toward_tm = -rel_teammate                                     # (B, 2)
        wide_target = toward_tm + torch.tensor(
            [0.2, self.wide_offset], device=obs.device
        )
        dir_to_wide = _unit(wide_target)

        # Blend: mostly wide positioning, some goal bias.
        support_action = _unit(0.7 * dir_to_wide + 0.3 * dir_to_goal)

        # Fallback to direct ball-chase when far from ball.
        too_far = dist_to_ball > self.fallback_dist
        return torch.where(too_far, dir_to_ball, support_action)


class ManMarkerPolicy(torch.nn.Module):
    """Man-to-man marking defender: shadows the opponent goal-side of them.

    Holds a position between the single opponent and the ego's own goal so the
    opponent cannot cut inside for an open shot or pass. The standoff distance
    tightens once the opponent enters the defensive third near the ego's own
    goal, and the policy contests the opponent directly at very close range.

    Both standoff distances must stay comfortably inside the scenario's
    `neutralize_radius` (0.15), or this policy fails to actually freeze the
    opponent and stops being a marking partner at all: at the original 0.18 it
    held red frozen only 26% of the time.
    """

    goal_offset: float = 3.01  # red_goal.x - blue_goal.x, used to derive own-goal-relative vectors
    marking_distance: float = 0.09
    tight_marking_distance: float = 0.05
    danger_zone_x: float = 1.0      # opponent within this x-distance of own goal is "dangerous"
    challenge_radius: float = 0.05  # contest the opponent directly once this close

    def __init__(self, env=None):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rel_goal = obs[:, 12:14]  # agent_pos - red_goal_pos
        rel_adv = obs[:, 16:18]   # agent_pos - opponent_pos

        rel_blue_goal = rel_goal + torch.tensor([self.goal_offset, 0.0], device=obs.device)
        opp_rel_blue_goal = rel_blue_goal - rel_adv  # opponent_pos - blue_goal_pos

        dir_opp_to_own_goal = _unit(-opp_rel_blue_goal)
        dangerous = opp_rel_blue_goal[:, 0:1] < self.danger_zone_x
        standoff = torch.where(
            dangerous,
            torch.tensor(self.tight_marking_distance, device=obs.device),
            torch.tensor(self.marking_distance, device=obs.device),
        )

        # Goal-side shadow position: offset from the opponent toward own goal.
        target_rel = rel_adv - standoff * dir_opp_to_own_goal
        dir_to_target = _unit(-target_rel)

        dist_to_opp = torch.linalg.norm(rel_adv, dim=-1, keepdim=True)
        dir_to_opp = _unit(-rel_adv)
        return torch.where(dist_to_opp < self.challenge_radius, dir_to_opp, dir_to_target)


# Held-out partners for zero-shot coordination evaluation.
#
# `role` is the role each partner commits to, which is what makes role adaptation
# measurable: a cooperator that coordinates takes the *complementary* role, so
# P(ego marks | role="attack") should differ sharply from P(ego marks | role="mark").
# A partner-independent policy gives the same number for both.
#
# Measured against an idle ego under rl_cfg/football.yaml (128 episodes):
#   name             return   red frozen%   ball contact%   blue score%
#   scripted          13.87        42.4             2.1            1.6
#   direct_attacker   62.52        10.6            31.1           67.2
#   wide_support      -7.97         7.3             0.1            0.0
#   man_marking       11.05        73.4             0.2            0.0
EVALUATION_BEHAVIOR_SET: list[dict[str, Any]] = [
    {
        "name": "scripted",
        "description": "Hybrid scripted policy with dribbling, passing, and shooting logic. Yields possession "
        "whenever the teammate is closer to the ball, so it commits to neither role and is excluded from the "
        "role-adaptation conditioning.",
        "policy_cls": ScriptedPolicy,
        "role": "mixed",
    },
    {
        "name": "direct_attacker",
        "description": "Pushes the ball toward goal by approaching from behind it, blending continuously "
        "between circling to get goal-side of the ball and driving through it. Ignores teammates. Holds the "
        "ball 31% of steps and scores 67% of episodes unaided, so the complementary role is to mark.",
        "policy_cls": DirectAttackerPolicy,
        "role": "attack",
    },
    {
        "name": "wide_support",
        "description": "Drifts to a wide support position to provide a passing lane rather than chasing the ball. "
        "Touches the ball ~0% of steps and never neutralises the opponent, so it fills neither role and leaves "
        "both to the cooperator.",
        "policy_cls": WideSupportPolicy,
        "role": "none",
    },
    {
        "name": "man_marking",
        "description": "Man-to-man marks the opponent, shadowing them goal-side inside the neutralisation radius. "
        "Holds the opponent frozen 73% of steps and never touches the ball, so the complementary role is to attack.",
        "policy_cls": ManMarkerPolicy,
        "role": "mark",
    },
]


def get_evaluation_scripted_policies() -> list[Callable[..., torch.nn.Module]]:
    """Return scripted policy factories for evaluation behavior sweeps."""
    return [entry["policy_cls"] for entry in EVALUATION_BEHAVIOR_SET]


def get_evaluation_partner_roles() -> dict[str, str]:
    """Map held-out partner name -> committed role ("attack" / "mark" / "none" / "mixed").

    Used to group evaluation rollouts when measuring whether the cooperator takes the
    role its partner left open.
    """
    return {entry["name"]: entry["role"] for entry in EVALUATION_BEHAVIOR_SET}