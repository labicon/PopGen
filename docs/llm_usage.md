# LLM Usage

This repo uses LiteLLM for all model calls. Configuration is centralized in `popgen/utils/config.py` via `LitellmConfig`.

## Where Models Are Used

1. Behavior generation strategies
- Config path: `behavior.model_config.model_name`
- Used by `generate_behaviors.py`

2. Agent/reward/policy generation
- Config path: `generation.model_name`
- Used by `generate_agents.py`

3. Evaluation (LLM-as-judge)
- Config path: `evaluation.evaluator_model`
- Used by adherence/quality/diversity evaluators

## `LitellmConfig`

`LitellmConfig` fields:
- `model_name: str`
- `api_base: str | None`
- `generation_kwargs: dict[str, Any]`

Example:

```python
from popgen.utils.config import LitellmConfig

cfg = LitellmConfig(
    model_name="openrouter/google/gemini-3-flash-preview",
    generation_kwargs={"temperature": 0.2},
)
```

## Provider/Auth

Set provider credentials in `.env` (repo root), then run scripts normally.

Typical examples:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- custom `api_base` for self-hosted endpoints

## Hydra Usage

Population harness:

```bash
cd scripts/population
python generate_agents.py \
  env=lbf \
  generation=policy_code \
  generation.model_name=openrouter/google/gemini-3-flash-preview \
  evaluation.evaluator_model=openrouter/google/gemini-3-flash-preview \
  behavior_path=./data/behaviors/lbf.json
```

LLM agent (the LLM acts as the policy at every timestep):

```bash
cd scripts/population
python generate_agents.py \
  env=lbf \
  generation=llm_agent \
  generation.model_name=openrouter/google/gemini-3-flash-preview \
  evaluation.evaluator_model=openrouter/google/gemini-3-flash-preview \
  behavior_path=./data/behaviors/lbf.json
```

## Notes

- `rl_reward` generation also consumes env-provided `REWARD_API_REFERENCE` to reduce hallucinated reward API usage.
- If you need provider-specific parameters, pass them through `generation_kwargs` in code or extend config plumbing in the harness script.
