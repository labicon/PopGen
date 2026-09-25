# Cooperator Training (Simplified)

This module trains a cooperator from a single `results_dir`.

## Input Contract

You provide:
- `env_module`
- `results_dir`
- `output_dir`

The pipeline:
1. Scans `behavior_*/attempt_*`.
2. Picks best attempt per behavior by max `evaluation.json -> overall_metrics.adherence`
   (tie: latest attempt index).
3. Loads teammate policy from:
   - `policy.py` -> `PolicyCode`, or
   - `checkpoints/checkpoint_*.pt` -> `RewardFn` checkpoint.
4. Trains cooperator with selected teammates.

## Public API

- `CooperatorTrainingPipeline`
- `select_policy_refs_from_behavior_results(results_dir)`

## Usage

```python
import importlib
from pathlib import Path
from popgen.cooperator import CooperatorTrainingPipeline

env_module = importlib.import_module("popgen.envs.point_mass_rendezvous.utils")

pipeline = CooperatorTrainingPipeline(
    env_module=env_module,
    output_dir=Path("scripts/cooperator/results/run1"),
    results_dir=Path("scripts/population/results/lbf/policy_code/<model>"),
)
experiment = pipeline.train_cooperator()
```
