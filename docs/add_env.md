# Adding a New Environment

Each environment lives in `popgen/envs/<env_name>/` with two files: `__init__.py` and `utils.py`.

## 1. Create `utils.py`

Define environment constants (descriptions, reward examples) and implement state extraction, rollout, and training functions:

```python
# popgen/envs/football/utils.py

from pathlib import Path
from benchmarl.environments import VmasTask
from pydantic import BaseModel, Field
from vmas.scenarios import football as football_scenario
from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

RL_CFG_DIR = Path(__file__).parent / "rl_cfg"

ENVIRONMENT_DESCRIPTION = """You are playing a 2v2 football game.
You are in the blue team. You must coordinate with a teammate to score.
..."""

REWARD_EXAMPLE = """
def reward_fn(self, agent: Agent):
    reward = torch.zeros(self.world.batch_dim, device=self.world.device)
    blue_score = over_right_line * goal_mask
    red_score = over_left_line * goal_mask
    return self.scoring_reward * blue_score - self.scoring_reward * red_score
"""

# ---------------------------------------------------------------------------- #
#                                 Action Type                                  #
# ---------------------------------------------------------------------------- #

class ActionType(BaseModel):
    """Action type for the football environment."""
    fx: float = Field(description="Horizontal force component (-1=left, 1=right)")
    fy: float = Field(description="Vertical force component (-1=down, 1=up)")

    @classmethod
    def default_action(cls):
        return cls(fx=0.0, fy=0.0)

    def to_tensor(self):
        return torch.tensor([self.fx, self.fy], dtype=torch.float32)

# ---------------------------------------------------------------------------- #
#                              State Extraction                                #
# ---------------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #

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

## 2. Create `__init__.py`

Export all components:

```python
# popgen/envs/football/__init__.py

from .utils import ENVIRONMENT_DESCRIPTION, REWARD_EXAMPLE, ActionType, rollout_fn, train_fn
```

## 3. Run

```bash
python scripts/behavior_policy/main.py env=football
```
