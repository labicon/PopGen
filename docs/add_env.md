# Adding a New Environment

Each environment lives in `popgen/envs/<env_name>/` with three files: `__init__.py`, `prompts.py`, and `utils.py`.

## 1. Create `prompts.py`

Define LLM prompts that describe the environment:

```python
# popgen/envs/football/prompts.py

ENVIRONMENT_DESCRIPTION = """You are playing a 2v2 football game.
You are in the blue team. You must coordinate with a teammate to score.

When you want to describe agent behaviors as code, these are the accessible variables:
- `ego_agent_position`: A tuple (x, y) representing the position of ego agent.
- `teammate_position`: A tuple (x, y) representing the position of teammate.
- `adversary_positions`: A list of tuples [(x, y), ...] for adversary positions.
- `ball_position`: A tuple (x, y) representing the ball position.
- `goal_positions`: A tuple (x, y) representing the goal position."""

ACTION_TYPE = "Continuous"

ACTION_FORMAT = """The action should be a torch tensor of shape (1, 2) representing force.
Example: `torch.tensor([[1.0, 0.5]])` moves right with force 1.0, up with force 0.5."""

OBSERVATION_FORMAT = """
def observation(self, agent: Agent):
    obs = {
        "obs": [agent_force, agent_pos - ball_pos, agent_vel - ball_vel, ...],
        "pos": [agent_pos - goal_pos],
        "vel": [agent_vel],
        "adversaries": [...],
        "teammates": [...],
    }
    return torch.cat(list(obs.values()), dim=-1)
"""

EGO_VISUAL_DESCRIPTION = """Ego team wears blue jerseys on the left side.
The ego agent is the blue player with tag #1."""

REWARD_FORMAT = """
def reward_fn(self, agent: Agent):
    reward = torch.zeros(self.world.batch_dim, device=self.world.device)
    blue_score = over_right_line * goal_mask
    red_score = over_left_line * goal_mask
    return self.scoring_reward * blue_score - self.scoring_reward * red_score
"""
```

## 2. Create `utils.py`

Implement state extraction, rollout, and training functions:

```python
# popgen/envs/football/utils.py

from pathlib import Path
from benchmarl.environments import VmasTask
from vmas.scenarios import football as football_scenario
from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

RL_CFG_DIR = Path(__file__).parent / "rl_cfg"


def get_state_football(env):
    """Extract state for logging."""
    state = {}
    for agent in env.scenario.blue_agents:
        state[agent.name] = {
            "pos": [round(x, 2) for x in agent.state.pos[0].cpu().numpy().tolist()],
            "vel": [round(x, 2) for x in agent.state.vel[0].cpu().numpy().tolist()],
        }
    state["env"] = {
        "ball_pos": [round(x, 2) for x in env.scenario.ball.state.pos[0].cpu().numpy().tolist()],
    }
    return state


def rollout_fn(policy, env=None, **kwargs):
    if env is None:
        env = VmasTask.FOOTBALL.get_from_yaml(path=str(RL_CFG_DIR / "football.yaml")).get_env_fun(
            num_envs=1, continuous_actions=True, seed=0, device="cpu"
        )()
    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 200
    return rollout_vmas_env(policy=policy, env=env, get_state_fn=get_state_football, **kwargs)


def train_fn(reward_fn: callable, output_dir: str, **kwargs):
    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=football_scenario,
        scenario_name="football",
        vmas_task=VmasTask.FOOTBALL,
        rl_cfg_dir=RL_CFG_DIR,
        **kwargs,
    )
```

## 3. Create `__init__.py`

Export all components:

```python
# popgen/envs/football/__init__.py

from .prompts import (
    ACTION_FORMAT, ACTION_TYPE, EGO_VISUAL_DESCRIPTION,
    ENVIRONMENT_DESCRIPTION, OBSERVATION_FORMAT, REWARD_FORMAT
)
from .utils import rollout_fn, train_fn

__ALL__ = [
    "ENVIRONMENT_DESCRIPTION", "ACTION_FORMAT", "OBSERVATION_FORMAT",
    "EGO_VISUAL_DESCRIPTION", "REWARD_FORMAT", "ACTION_TYPE",
    "rollout_fn", "train_fn",
]
```

## 4. Run

```bash
python scripts/behavior_policy/main.py env=football
```
