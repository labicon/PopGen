# Population Scripts

This folder contains:
- `main.py`: run behavior generation + policy/reward/agent generation + evaluation.
- `gather_results.py`: aggregate run outputs into one CSV.

## Prerequisites

- Run from repo root: `zeroshot_coordination/`
- Install project dependencies.
- Set required API keys in `.env` (loaded by `main.py` via `python-dotenv`).

## Run `main.py`

Default run:

```bash
python scripts/population/main.py
```

This uses config from `scripts/population/cfg/config.yaml`:
- `generation=policy_code`
- `behavior=verbalized_sampling`
- `env=lbf`

### Common overrides

Switch generation type:

```bash
python scripts/population/main.py generation=llm_agent
python scripts/population/main.py generation=rl_reward
```

Switch behavior strategy:

```bash
python scripts/population/main.py behavior=in_context
python scripts/population/main.py behavior=stratified
python scripts/population/main.py behavior=verbalized_sampling_multiturn
```

Set env, seeds, and scale:

```bash
python scripts/population/main.py env=football n_behaviors=10 n_workers=10 seed=42
```

Set a custom run name:

```bash
python scripts/population/main.py run_name=my_run
```

Run multiple trials:

```bash
python scripts/population/main.py num_trials=3
```

Reuse pre-generated behaviors:

```bash
python scripts/population/main.py behavior_path=/path/to/behaviors.json
```

### Output layout

Results are written under:

```text
results/<env>/<generation.type>/<behavior_source>/<run_name>/
```

`results/` is relative to the directory where you launch the command.

Inside each run:

```text
.hydra/config.yaml
behaviors.json
trial_<k>/behavior_<i>/attempt_<j>/...   # when num_trials > 1
behavior_<i>/attempt_<j>/...             # when num_trials == 1
```

Attempt folders contain:
- `evaluation.json`
- `latency.json`
- `model_outputs.jsonl`

## Run `gather_results.py`

Aggregate one or more experiment folders into one CSV.

```bash
python scripts/population/gather_results.py \
  scripts/population/results/lbf/policy_code/VerbalizedSamplingStrategy \
  -o /tmp/gathered_results.csv
```

Multiple experiment folders:

```bash
python scripts/population/gather_results.py \
  scripts/population/results/lbf/policy_code/VerbalizedSamplingStrategy \
  scripts/population/results/lbf/llm_agent/VerbalizedSamplingStrategy \
  -o /tmp/gathered_results.csv
```

### CSV columns

- `experiment_id`
- `run_id`
- `env`
- `seed`
- `generation_type`
- `generation_model`
- `trial_id`
- `behavior_id`
- `attempt_id`
- `rollout_idx`
- `success`
- `generation_tokens`
- `generation_latency`
- `rollout_latency`

### Notes

- `generation_tokens` is computed from `model_outputs.jsonl` using `tiktoken` (`o200k_base`).
- For `generation_type=llm_agent`, token counts align rollouts with the last entries in `model_outputs.jsonl`.
