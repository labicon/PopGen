# Population Harness

This directory contains the current single-behavior generation/evaluation harness.

## Scripts

- `generate_behaviors.py`: generate behavior descriptions into `data/behaviors/`
- `generate_agents.py`: run agent generation/evaluation for a behavior file
- `gather_results.py`: aggregate one or more result directories into CSV
- `run_pipeline.sh`: convenience wrapper for all three steps

## Quick Start

```bash
cd scripts/population
./run_pipeline.sh env=lbf generation=policy_code
```

## Step-by-Step

1. Generate behaviors:

```bash
python generate_behaviors.py env=lbf n_behaviors=5
```

2. Generate/evaluate agents:

```bash
python generate_agents.py \
  behavior_path=./data/behaviors/lbf.json \
  env=lbf \
  generation=policy_code
```

3. Gather results:

```bash
python gather_results.py \
  ./results/lbf/policy_code/gemini-3-flash-preview \
  -o ./results/lbf/policy_code/gemini-3-flash-preview/results.csv
```

## Parallelism

- `n_workers`: number of jobs launched in parallel (outer level)
- `pipeline_workers`: workers used inside each run for rollout/eval internals (inner level, defaults to `1`)

Effective load is roughly multiplicative (`n_workers * pipeline_workers`).

## Defaults

See `cfg/config.yaml` for current defaults:
- `n_trials`
- `n_workers`
- `evaluation.n_eval_rollouts`
- `evaluation.max_alignment_retries`

Generation-specific defaults live under `cfg/generation/*.yaml`.

## Output Layout

```text
<output_dir>/
  agents/
    behavior_<i>/
      <run_timestamp>/
        attempt_<j>/
          evaluation.json
          latency.json
          model_outputs.jsonl
          rollouts/...
  metadata/
    <timestamp>/
      behaviors.json
      summary.json
```

`gather_results.py` supports this layout and the legacy layout.
