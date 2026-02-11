from pathlib import Path

import torch
from benchmarl.environments import VmasTask
from pydantic import BaseModel, Field
from vmas.scenarios import point_mass_rendezvous as point_mass_rendezvous_scenario

from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

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
- Index 4-11: `landmark_positions` - A list of tuples [(x, y), ...] representing the positions of the 4 landmarks.
- Index 12-13: `other_agent_relative_position` - A tuple (x, y) representing the position of the other agent relative to the ego agent (other_agent_position - ego_agent_position).

> **Action Space**:
The agents can apply a force in the horizontal and vertical directions to move around the space.
The action is represented as a tuple (fx, fy) where:
- `fx`: Horizontal force component, where -1 means full force to the left and 1 means full force to the right.
- `fy`: Vertical force component, where -1 means full force downwards and 1 means full force upwards."""

REWARD_EXAMPLE = """Simple sparse reward: +1 if both agents are on the same landmark at the same time step, otherwise 0.

```python
def reward_fn(self, agent: Agent):

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
#                              State Extraction                                #
# ---------------------------------------------------------------------------- #


def get_state_point_mass_rendezvous(env):
    state = {}

    def _get_agent_state(agent):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        action = agent.action.u[0].cpu().numpy().tolist()
        return {
            "action": [round(x, 2) for x in action],
            "pos": [round(x, 2) for x in pos],
            "vel": [round(x, 2) for x in vel],
        }

    for i, agent in enumerate(env.world.agents):
        state[agent.name] = _get_agent_state(agent)

    for i, landmark in enumerate(env.world.landmarks):
        pos = landmark.state.pos[0].cpu().numpy().tolist()
        state[f"landmark{i}_pos"] = [round(x, 2) for x in pos]

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
    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=point_mass_rendezvous_scenario,
        scenario_name="point_mass_rendezvous",
        vmas_task=VmasTask.POINT_MASS_RENDEZVOUS,
        rl_cfg_dir=RL_CFG_DIR,
        **kwargs,
    )


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy loaded from the checkpoint in save_dir.
    Uses the experiment's test_env to ensure correct observation structure.
    """
    # if env is not defined, load default pm_rdv environment
    if env is None:
        env = VmasTask.POINT_MASS_RENDEZVOUS.get_from_yaml(path=str(RL_CFG_DIR / "point_mass_rendezvous.yaml")).get_env_fun(
            num_envs=1,
            continuous_actions=True,
            seed=0,
            device="cpu",
        )()

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 100  # Default max steps for point_mass_rendezvous

    return rollout_vmas_env(policy=policy, env=env, get_state_fn=get_state_point_mass_rendezvous, **kwargs)
