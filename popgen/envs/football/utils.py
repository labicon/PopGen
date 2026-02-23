from pathlib import Path

import yaml
import torch
from benchmarl.environments import VmasTask
from pydantic import BaseModel, Field
from vmas.scenarios import football as football_scenario

from popgen.agents.utils import extract_observation_and_action_key, format_action_output
from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

ENVIRONMENT_DESCRIPTION = """The environment is a football (soccer) game in a simulated environment.
The objective of the blue team is to score a goal by getting the ball into the red team's goal area while preventing the red team from scoring in the blue team's goal area.

Each team consists of 2 agents.
The agents can move around the field and kick the ball by applying forces to it.
The agents must coordinate with their teammate to score goals and defend against the opponent team.

Note that the ego team is located in the left side of the football field, wearing blue jerseys, and the opponent team is located in the right side of the football field, wearing red jerseys. The ego agent is represented as the blue football player with the tag #1.

> **Observation Space**:
Each agent observes a flattened vector describing its interaction with the ball, goal, teammate, and adversaries. All vectors are typically 2D `(x, y)`.
- Index 0–1: `ego_agent_force` — Current force applied to the ego agent (captures intended acceleration).
- Index 2–3: `ego_to_ball_relative_position` — Direction and distance from the ball to the agent.
- Index 4–5: `ego_to_ball_relative_velocity` — Whether the agent is closing in on or moving away from the ball.
- Index 6–7: `ball_to_goal_relative_position` — How far and in what direction the ball is from the goal.
- Index 8–9: `ball_velocity` — Speed and direction of the ball's movement.
- Index 10–11: `ball_force` — Current force affecting the ball (e.g., from recent contacts).
- Index 12–13: `ego_to_goal_relative_position` — Agent's positioning relative to the scoring target.
- Index 14–15: `ego_velocity` — Agent's own movement speed and direction.

*Adversaries Elements (each contributes 8 dims):*
- Index 16–23: `adversary_0_features` — Relative position and velocity to adversary 0 plus its absolute velocity and force; helps anticipate opponent behavior.
- Index 24–31: `adversary_1_features` — Same information for adversary 1.

*Teammates Elements (each contributes 8 dims):*
- Index 32–39: `teammate_0_features` — Relative position and velocity to the teammate plus their velocity and force; useful for coordination and spacing.

> **Action Space**:
The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.
The action is represented as a tuple (fx, fy) where:
- `fx`: Horizontal force component, where -1 means full force to the left and 1 means full force to the right.
- `fy`: Vertical force component, where -1 means full force downwards and 1 means full force upwards.
Note that you can only move the agent, and therefore do not have direct access to kick the ball. You should move towards the ball and apply force to it in order to kick the ball towards the goal."""

REWARD_EXAMPLE = """Simple reward function example for the football environment:

```python
def reward_fn(self, agent: Agent):

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
#                              State Extraction                                #
# ---------------------------------------------------------------------------- #


def get_state_football(env):
    state = {}

    blue_team, red_team = env.scenario.blue_agents, env.scenario.red_agents
    goal_pos = env.scenario.right_goal_pos

    def _get_agent_state(agent):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        action = agent.action.u[0].cpu().numpy().tolist()
        return {
            "action": [round(x, 2) for x in action],
            "pos": [round(x, 2) for x in pos],
            "vel": [round(x, 2) for x in vel],
        }

    # Ego team agents
    for agent in blue_team:
        state[agent.name] = _get_agent_state(agent)

    # Adversary team agents
    for agent in red_team:
        state[agent.name] = _get_agent_state(agent)

    # Goal position
    env_state = {}
    env_state["goal_pos"] = [round(x, 2) for x in goal_pos.cpu().numpy().tolist()]

    # Ball state
    ball_pos = env.scenario.ball.state.pos[0].cpu().numpy().tolist()
    ball_vel = env.scenario.ball.state.vel[0].cpu().numpy().tolist()
    env_state["ball_pos"] = [round(x, 2) for x in ball_pos]
    env_state["ball_vel"] = [round(x, 2) for x in ball_vel]

    state["env"] = env_state

    return state


# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #


def train_fn(reward_fn: callable, output_dir: str, **kwargs):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward tensor.
        output_dir: Directory to save results.
    """
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

    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=football_scenario,
        scenario_name="football",
        vmas_task=VmasTask.FOOTBALL,
        rl_cfg_dir=RL_CFG_DIR,
        teammate_policy=[scripted_policy],
        **kwargs,
    )


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy loaded from the checkpoint in save_dir.
    Uses the experiment's test_env to ensure correct observation structure.
    """
    # if env is not defined, load default football environment
    if env is None:
        env = create_env(**kwargs)

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 200  # Default max steps for football

    # If the policy is list of policies, wrap them with MultiPolicyWrapper to create a single policy that can be used for rollout.
    if isinstance(policy, list) and len(policy) > 1:
        from popgen.agents.scripted_partner import MultiPolicyWrapper

        policy = MultiPolicyWrapper(policies=policy)

    return rollout_vmas_env(
        policy=policy,
        env=env,
        get_state_fn=get_state_football,
        **kwargs,
    )

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
    return VmasTask.FOOTBALL.get_from_yaml(path=str(RL_CFG_DIR / "football.yaml")).get_env_fun(
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
    )()

# ---------------------------------------------------------------------------- #
#                              Scripted Policy                                 #
# ---------------------------------------------------------------------------- #


class ScriptedPolicy(torch.nn.Module):
    """Simple scripted football partner policy.

    Moves toward the ball direction inferred from the observation:
    obs[2:4] = ego_to_ball_relative_position.
    """

    def __init__(self, env=None):
        super().__init__()
        self.env = env
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs) -> torch.Tensor:
        obs_tensor, action_key = extract_observation_and_action_key(obs)
        # Use `...` to index the last (obs feature) dimension regardless of
        # whether obs_tensor is [batch, obs_dim] or [batch, 1, obs_dim].
        direction_to_ball = -obs_tensor[..., 2:4]
        norm = torch.linalg.norm(direction_to_ball, dim=-1, keepdim=True) + 1e-8
        action = (direction_to_ball / norm).clamp(-1, 1)
        return format_action_output(obs, action_key, action)
