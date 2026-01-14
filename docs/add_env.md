# Adding a New Environment

This guide explains how to add a new environment to the repository.

## Directory Structure

Each environment lives in `popgen/envs/<env_name>/` with the following structure:

```
popgen/envs/<env_name>/
├── __init__.py
├── prompts.py
└── rollout.py
```

## Step 1: Create the Environment Folder

```bash
mkdir -p popgen/envs/<env_name>
```

## Step 2: Create `prompts.py`

This file contains the LLM prompts that describe the environment. Define three constants:

```python
ENVIRONMENT_DESCRIPTION = """
<Describe the environment, task objectives, agents, and their roles.>
<List accessible variables for behavior code generation.>
"""

OBSERVATION_FORMAT = """
<Provide the observation function code or description.>
<This helps the LLM understand how observations are structured.>
"""

EGO_VISUAL_DESCRIPTION = "<Short description of how the ego agent appears visually.>"
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
```

## Step 3: Create `rollout.py`

This file contains the rollout logic. You need to implement two functions:

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
def rollout_fn(agent0_policy: callable, agent1_policy: callable):
    """
    Execute a rollout with the given policies.

    Args:
        agent0_policy: Policy function for agent 0. Takes observation, returns action.
        agent1_policy: Policy function for agent 1. Takes observation, returns action.

    Returns:
        states: List of state dictionaries from each timestep.
        images: List of rendered frames (numpy arrays).
        error: Error message string if rollout failed, None otherwise.
    """
    max_steps = 200

    # Create environment
    env = make_env(scenario="<env_name>", ...)

    states = []
    images = []

    # Reset
    obs = env.reset()
    images.append(env.render(mode="rgb_array"))
    states.append(get_state_<env_name>(env))

    step = 0
    while step < max_steps:
        # Get actions from policies
        try:
            action_0 = agent0_policy(obs[0].squeeze(0))
            action_1 = agent1_policy(obs[1].squeeze(0))
        except Exception as e:
            return states, images, str(e)

        # Step environment
        obs, rews, dones, info = env.step([action_0, action_1])

        # Record state and image
        states.append(get_state_<env_name>(env))
        images.append(env.render(mode="rgb_array"))

        step += 1
        if dones.any():
            break

    return states, images, None
```

**Important**: The `rollout_fn` signature should match the policy functions that the LLM will generate. If your environment has more agents, adjust accordingly.

## Step 4: Create `__init__.py`

Export all required components:

```python
from .prompts import ENVIRONMENT_DESCRIPTION, OBSERVATION_FORMAT, EGO_VISUAL_DESCRIPTION
from .rollout import rollout_fn

__ALL__ = ["ENVIRONMENT_DESCRIPTION", "OBSERVATION_FORMAT", "EGO_VISUAL_DESCRIPTION", "rollout_fn"]
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
- [ ] Created `prompts.py` with `ENVIRONMENT_DESCRIPTION`, `OBSERVATION_FORMAT`, `EGO_VISUAL_DESCRIPTION`
- [ ] Created `rollout.py` with `get_state_<env_name>()` and `rollout_fn()`
- [ ] Created `__init__.py` exporting all components
- [ ] Tested the environment with `python scripts/behavior_policy/main.py env=<env_name>`
