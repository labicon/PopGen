# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PopGen** is a framework for generating diverse teammate populations for multi-agent reinforcement learning (zero-shot coordination). It provides three generation pipelines:
1. **LLM agents** – configure LLMs to act as real-time agents
2. **Policy code** – generate Python functions as deterministic policies
3. **Reward functions** – generate reward functions, then train RL agents via BenchMARL

## Commands

### Install
```bash
uv sync              # base dependencies
uv sync --extra dev  # include pytest and ruff
```
Deps must also be installed separately:
```bash
pip install -e deps/VectorizedMultiAgentSimulator
pip install -e deps/BenchMARL
pip install -e deps/lb-foraging
```

### Run tests
```bash
uv run pytest tests/ -v
uv run pytest tests/generation/test_policy_generator.py -v  # single test file
```

### Lint / format
```bash
uv run ruff check .
uv run ruff format .
```

### Run scripts (Hydra-based)
```bash
# Population generation (from scripts/population/): behaviors → agents → CSV
./run_pipeline.sh env=lbf generation=policy_code        # or generation=rl_reward / llm_agent
python generate_behaviors.py env=lbf behavior=verbalized_sampling
python generate_agents.py env=lbf behavior_path=./data/behaviors/lbf.json

# Cooperator (from scripts/cooperator/)
./run_experiments.sh env=lbf n_behaviors=8               # population + cooperator + evaluation
./run_cooperator.sh env=lbf population_dir=/path/to/population
python generate_cooperator.py population.env=lbf cooperator.results_dir=/path/to/population
python gather_results.py --cooperator_dir /path/to/cooperator --env lbf --n_eval_rollouts 50

# Baselines
cd scripts/self-play && ./run_experiments.sh env=pp_stag
cd scripts/fcp && python main.py
```

## Architecture

### Three-stage pipeline
The scripts in `scripts/population/` run three conceptual stages:
1. **Behavior generation** – produce a diverse set of `Behavior` objects (name + description) using a `GenerationStrategy`
2. **Policy/reward generation** – convert each `Behavior` into executable policy code, a reward function (then RL-trained), or an LLM-agent config
3. **Evaluation** – run rollouts, score adherence/diversity/quality

The core abstractions live in `popgen/population/`:
- `pipeline.py` – `BaseGenerator`, `GeneratorWithFeedback`
- `types.py` – `Behavior` (Pydantic model)
- `generation/diversity/` – strategies: `DirectStrategy`, `StratifiedStrategy`, `VerbalizedSamplingStrategy`
- `evaluation/` – `BehaviorAdherenceEvaluator`, `SimilarityEvaluator`, `DiversityEvaluator`

### Agent types (`popgen/agents/`)
- `PolicyCodeAgent` – wraps a generated Python function; extends `nn.Module` for VMAS/BenchMARL compatibility
- `LLMAgent` – calls an LLM at each timestep to produce actions
- `MultiPolicyRouter` / `EgoTeammateRouter` – route observations to ego and teammate policies
- `callbacks.py` – BenchMARL callbacks for population/teammate sampling

### Environment structure (`popgen/envs/`)
Each environment directory (e.g., `lbf/`, `point_mass_rendezvous/`, `football/`) exposes:
- `ENVIRONMENT_DESCRIPTION` – natural language description fed to LLMs
- `REWARD_EXAMPLE` – example reward function for few-shot prompting
- `ActionType` – Pydantic model for the action space (used by `LLMAgent`)
- `rollout_fn()` – run a policy for evaluation
- `train_fn()` – train via BenchMARL (reward-generation pipeline)
- `ScriptedPolicy` – reference policy used during code generation/testing

Shared utilities: `popgen/envs/vmas_common.py` (VMAS environments), `popgen/envs/lbf_common.py` (LBF).

### LLM interface (`popgen/utils/config.py`)
All LLM calls use `LitellmConfig` (wrapping LiteLLM), enabling transparent switching between OpenAI, Anthropic, vLLM, etc. Model names follow LiteLLM conventions (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3-5-haiku-20241022`).

### Configuration (Hydra)
All scripts use Hydra with YAML configs under `cfg/` next to each script. Key config fields: `env`, `behavior` (strategy), `generator.model_name`, `max_behaviors`, `n_workers`, `adherence_threshold`. Results are saved to `results/${env}/...`.

### Cooperator training (`popgen/cooperator/`, `scripts/cooperator/`)
A fourth stage on top of the three-stage pipeline: train a *cooperator* (ego agent) against the generated population.

- `CooperatorTrainingPipeline` – scans a population `results_dir`, picks the best attempt per behavior by adherence score, loads teammate policies (`PolicyCode` → `PolicyCodeAgent`, `RewardFn` → BenchMARL checkpoint), and trains the ego via `env_module.train_fn`.
- `select_policy_refs_from_behavior_results(results_dir)` – returns a list of `PolicyArtifactRef` (kind + path + metadata) for the best attempts found.
- `scripts/cooperator/population_cfg/` – Hydra config for generating the population from this directory (mirrors `scripts/population/` but scoped to the cooperator workflow).
- `scripts/cooperator/generate_cooperator.py` – Hydra entry point; reads `population.env` + `cooperator.results_dir` and calls `CooperatorTrainingPipeline`.
- `scripts/cooperator/gather_results.py` – standalone re-evaluation: loads a cooperator checkpoint and runs rollouts against the `ScriptedPolicy` and the saved policy-code population.
- `scripts/self-play/`, `scripts/fcp/` – self-play and FCP baselines.

## Environment Setup

API keys must be set in a `.env` file (see `.env.example`):
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Adding a New Environment

See `docs/add_env.md`. In brief: implement `ENVIRONMENT_DESCRIPTION`, `rollout_fn`, `train_fn`, `ScriptedPolicy` in `popgen/envs/<name>/__init__.py`; envs are loaded by name via `importlib`.

## Code Style

- Python 3.12+, managed with `uv`
- Ruff: 200-char line length; `E`, `W`, `F`, `I`, `UP` rules; `F401`/`F403` ignored
- Environment files (`popgen/envs/**/*.py`) are excluded from Ruff
