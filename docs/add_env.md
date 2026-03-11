# Adding a New Environment

Environment modules live in `popgen/envs/<env_name>/`.

Current pipeline contract expects:
- `popgen/envs/<env_name>/utils.py`
- `popgen/envs/<env_name>/__init__.py` exporting the required symbols

## Required Exports

From `utils.py`, define and export:

1. `ENVIRONMENT_DESCRIPTION: str`
- Natural-language environment description used in behavior generation and evaluation prompts.
- Should clearly identify the ego agent in renderings (color/marker/index), and include default team sizes plus observation/action conventions.

2. `REWARD_EXAMPLE: str`
- Example reward function snippet used to guide `rl_reward` generation.

3. `REWARD_API_REFERENCE: str` (recommended)
- Explicit reference of callable attributes/state available to generated reward code.
- This is injected into the reward-generation prompt.

4. `ActionType`
- Pydantic model used by `llm_agent` action parsing.

5. `create_env(...)`
- Factory that returns an environment instance.

6. `rollout_fn(policy, env=None, **kwargs)`
- Must return `(rollout_data, images, error)`.
- `rollout_data` should be serializable and suitable for evaluation.

7. `train_fn(reward_fn, output_dir, **kwargs)`
- Used by `rl_reward` generation to train a policy from generated reward code.

8. `ScriptedPolicy`
- Teammate policy used during rollouts/training where applicable.

9. `serialize_state_fn(rollout_data: list[dict]) -> list[dict]`
- Converts raw observation vectors into human-readable structured state dicts for LLM-based evaluation.
- Each agent entry in the rollout data gains a `serialized_state` key containing named fields (positions, distances, zones, etc.) derived from the raw `state` vector.
- Used by the adherence evaluator to provide the judge model with interpretable state descriptions instead of opaque numeric tensors.

## Rollout State Format

For compatibility with evaluator/logging utilities, rollout state entries should follow:

```python
{
  "step": 0,
  "ego_team": {
    "ego_agent": {"state": [...], "action": [...], "reward": ...},
    "teammate_0": {"state": [...], "action": [...], "reward": ...},
  },
  "opponent_team": {
    "opponent_0": {"state": [...], "action": [...], "reward": ...},
  },
  "env": {"key": "value"},
}
```

Notes:
- The canonical naming expected by downstream evaluators is `ego_team` / `opponent_team` with `ego_agent` as the controlled agent key.
- The `step` field should be the integer timestep index.
- The `env` dict holds any environment-level metadata (e.g. scores, flags).

## `__init__.py` Template

```python
from .utils import (
    ENVIRONMENT_DESCRIPTION,
    REWARD_API_REFERENCE,
    REWARD_EXAMPLE,
    ActionType,
    ScriptedPolicy,
    create_env,
    rollout_fn,
    serialize_state_fn,
    train_fn,
)
```

## Minimal Validation

1. Generate behaviors:

```bash
cd scripts/population
python generate_behaviors.py env=<env_name> n_behaviors=2
```

2. Run one agent-generation config dump (no full rollout/training):

```bash
python generate_agents.py env=<env_name> behavior_path=./data/behaviors/<env_name>.json --cfg job
```

3. Optional end-to-end quick run:

```bash
python generate_agents.py env=<env_name> behavior_path=./data/behaviors/<env_name>.json n_behaviors=1 n_trials=1
```
