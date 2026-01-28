# Adding a New Environment

This guide explains how to add a new environment to the repository.

## Directory Structure

Each environment lives in `popgen/envs/<env_name>/` with the following structure:

```
popgen/envs/<env_name>/
├── __init__.py
├── prompts.py
├── utils.py
└── rl_cfg/ (optional, for VMAS/BenchMARL training configs)
```

## Step 1: Create the Environment Folder

```bash
mkdir -p popgen/envs/<env_name>
```

## Step 2: Create `prompts.py`

This file contains the LLM prompts that describe the environment. Define the constants below:

```python
ENVIRONMENT_DESCRIPTION = """
<Describe the environment, task objectives, agents, and their roles.>
<List accessible variables for behavior code generation.>
"""

ACTION_FORMAT = """
<Describe the action format expected by the policy generator.>
<Include tensor shapes and an example.>
"""

OBSERVATION_FORMAT = """
<Provide the observation function code or description.>
<This helps the LLM understand how observations are structured.>
"""

EGO_VISUAL_DESCRIPTION = "<Short description of how the ego agent appears visually.>"

REWARD_FORMAT = """
<Provide a template or description for compute_reward(self, agent).>
<Include available attributes and tensor shapes.>
"""
```

**Example** (from `point_mass_rendezvous`):

```python
ENVIRONMENT_DESCRIPTION = """
Two agents must coordinate to navigate to the same landmark simultaneously.
There are 4 landmarks available, representing different potential solutions.
...

When you want to describe agent behaviors as code, these are the accessible variables:
- `ego_agent_position`: A tuple (x, y) representing the position of ego agent.
- `other_agent_position`: A tuple (x, y) representing the position of other agent.
...
"""

OBSERVATION_FORMAT = """
def observation(self, agent: Agent):
    obs = [agent.state.pos, agent.state.vel]
    ...
    return torch.cat(obs, dim=-1)
"""

EGO_VISUAL_DESCRIPTION = "The ego agent is represented as the red point mass."

REWARD_FORMAT = """
def reward_fn(self, agent: Agent):
    reward = torch.zeros(self.world.batch_dim, device=self.world.device)
    ...
    return reward
"""
```

## Step 3: Create `utils.py`

This file contains the rollout and training helpers. You need to implement:

### 3.1 State Extraction Function

```python
def get_state_<env_name>(env):
    """Extract state information from the environment for logging."""
    state = {}
    # Extract relevant state information (positions, velocities, etc.)
    # Round values for readability
    state["agent0_pos"] = [round(x, 2) for x in pos]
    ...
    return state
```

### 3.2 Rollout Function

```python
def rollout_fn(policy: callable, env=None, **kwargs):
    """
    Execute a rollout with the given policies.

    Args:
        policy: Policy function for the ego agent. Takes observation, returns action.
        env: Optional pre-created environment instance.

    Returns:
        states: List of state dictionaries from each timestep.
        images: List of rendered frames (numpy arrays).
        error: Error message string if rollout failed, None otherwise.
    """
    # create env if needed, set defaults, then call shared helper
    if env is None:
        env = <create env here>
    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 100
    return rollout_vmas_env(policy=policy, env=env, get_state_fn=get_state_<env_name>, **kwargs)
```

### 3.3 Training Function (if you support reward learning)

```python
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
        scenario_module=<scenario_module>,
        scenario_name="<env_name>",
        vmas_task=<VMAS_TASK>,
        rl_cfg_dir=RL_CFG_DIR,
        **kwargs,
    )
```

**Important**: The `rollout_fn` signature should match the policy functions that the LLM will generate. If your environment has more agents, you can call the same `policy` for each agent or adapt the function accordingly.

## Step 4: Create `__init__.py`

Export all required components:

```python
from .prompts import ACTION_FORMAT, EGO_VISUAL_DESCRIPTION, ENVIRONMENT_DESCRIPTION, OBSERVATION_FORMAT, REWARD_FORMAT
from .utils import rollout_fn, train_fn

__ALL__ = [
    "ENVIRONMENT_DESCRIPTION",
    "ACTION_FORMAT",
    "OBSERVATION_FORMAT",
    "EGO_VISUAL_DESCRIPTION",
    "REWARD_FORMAT",
    "rollout_fn",
    "train_fn",
]
```

## Step 5: Run with the New Environment

Update the config or pass it as a command-line argument:

```yaml
# scripts/behavior_policy/cfg/config.yaml
env: <env_name>
```

Or run directly:

```bash
python scripts/behavior_policy/main.py env=<env_name>
```

## Checklist

- [ ] Created `popgen/envs/<env_name>/` folder
- [ ] Created `prompts.py` with `ENVIRONMENT_DESCRIPTION`, `ACTION_FORMAT`, `OBSERVATION_FORMAT`, `EGO_VISUAL_DESCRIPTION`, `REWARD_FORMAT`
- [ ] Created `utils.py` with `get_state_<env_name>()`, `rollout_fn()`, and optional `train_fn()`
- [ ] Created `__init__.py` exporting all components
- [ ] Tested the environment with `python scripts/behavior_policy/main.py env=<env_name>`
