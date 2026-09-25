# PopGen

PopGen generates and evaluates behavior-conditioned agents for multi-agent environments.

Pipelines:
- `scripts/population/` – behavior generation + per-behavior agent generation (policy code, RL reward, or LLM agent) and adherence evaluation
- `scripts/cooperator/` – train and evaluate a cooperator (ego agent) against a generated population
- `scripts/self-play/`, `scripts/fcp/` – self-play and FCP baselines

## Setup

Recommended:

```bash
uv sync
```

Alternative editable install:

```bash
pip install -e .
pip install -e deps/VectorizedMultiAgentSimulator
pip install -e deps/BenchMARL
```

Add provider keys in `.env` at repo root (copy from `.env.example` if present).

### SMACv2 Environment (StarCraft II)

SMACv2 in this repo uses the vendored package at `deps/smacv2` and a project-specific
installer script at `deps/install_smacv2.sh` (adapted from `deps/smacv2/README.md`).

From the repo root:

```bash
bash deps/install_smacv2.sh
pip install -e ./deps/smacv2
```

What this sets up:
- StarCraft II 4.10 under `deps/StarCraftII`
- SMAC maps under `deps/StarCraftII/Maps/SMAC_Maps`
- Local editable install for the vendored `smacv2` package from `deps/smacv2`

Notes:
- `deps/install_smacv2.sh` downloads large assets with `wget` and unpacks with `unzip`.
- The script is intended for Linux environments.
- If you need the vendored package's development extras, use `pip install -e "./deps/smacv2[dev]"`.

## Main Workflows

### Population harness

```bash
cd scripts/population
./run_pipeline.sh env=lbf generation=policy_code
```

Key scripts:
- `generate_behaviors.py`
- `generate_agents.py`
- `gather_results.py`

### Cooperator training

```bash
cd scripts/cooperator
./run_experiments.sh env=lbf n_behaviors=8          # population → cooperator → evaluation
./run_cooperator.sh env=lbf population_dir=/path/to/population   # cooperator on an existing population
```

### Baselines

```bash
cd scripts/self-play && ./run_experiments.sh env=pp_stag
cd scripts/fcp && python main.py
```

## Output Layout (Current)

The population harness produces per-experiment outputs with:

```text
<result_path>/
  agents/
    behavior_<i>/
      <run_timestamp>/
        attempt_<j>/...
  metadata/
    <timestamp>/
      behaviors.json
      summary.json
```

`gather_results.py` supports this layout and legacy layouts.

## Documentation

- Environment integration: `docs/add_env.md`
- LLM configuration/usage: `docs/llm_usage.md`
- Diversity discriminator evaluator: `docs/evaluators/diversity_discriminator.md`
