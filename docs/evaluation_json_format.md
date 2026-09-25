# Evaluation JSON Format

Every cooperator evaluator writes results under `<cooperator_dir>/evaluation/`.

For heldout scripted teammates, the evaluator uses fixed filenames:

| File | Evaluation mode |
|------|-----------------|
| `eval_teammates.json` | Standard scripted heldout teammates |

For population-backed evaluation (`policy_code` or `self_play`), the caller must provide an explicit `output_filename`. Common choices are:

| File | Evaluation mode |
|------|-----------------|
| `training_teammates.json` | In-domain training-population evaluation |
| `cross_eval_policy_code.json` | Cross-eval against a policy-code / reward-fn population |
| `cross_eval_self_play.json` | Cross-eval against a self-play checkpoint population |

Example:

```python
from popgen.cooperator import CooperatorEvaluator

summary = CooperatorEvaluator(
    cooperator_dir=cooperator_dir,
    env_module=env_module,
    eval_type="policy_code",
    population_dir=policy_code_pop_dir,
    num_workers=8,
).evaluate(
    n_eval_rollouts=50,
    output_filename="cross_eval_policy_code.json",
)
```

Baseline variants (produced by `BaselineEvaluator`) use the same schema and are written to `baseline_eval_teammates.json` and `baseline_training_teammates.json`.

---

## Top-level schema

All files share the same top-level structure:

```json
{
  "n_teammates": 4,
  "n_eval_rollouts": 50,
  "aggregate_reward_summary": { "mean": 12.3, "std": 1.1, "min": 10.0, "max": 14.0 },
  "aggregate_ego_reward_summary": { "mean": 4.5, "std": 0.7, "min": 3.2, "max": 5.4 },
  "aggregate_teammate_reward_summary": { "mean": 7.8, "std": 0.9, "min": 6.4, "max": 8.9 },
  "teammates": [<teammate entry>, "..."],
  "checkpoint_source": "latest"
}
```

| Key | Type | Description |
|-----|------|-------------|
| `n_teammates` | int | Number of teammates evaluated |
| `n_eval_rollouts` | int | Rollouts run per teammate |
| `aggregate_reward_summary` | object | Team-total reward stats aggregated across all rollouts of all teammates |
| `aggregate_ego_reward_summary` | object | Ego-agent reward stats aggregated across all rollouts of all teammates |
| `aggregate_teammate_reward_summary` | object | Summed teammate reward stats aggregated across all rollouts of all teammates |
| `teammates` | array | Per-teammate result entries |
| `checkpoint_source` | string | Always `"latest"` |

`aggregate_reward_summary` remains the canonical team-total field for backward compatibility.

---

## Per-teammate entry schema

Each element of `teammates` has this shape:

```json
{
  "idx": 0,
  "name": "ClosestFoodPolicy",
  "seed_dir": "/abs/path/to/seed_123",
  "policy_path": "/abs/path/to/policy.py",
  "n_rollouts": 50,
  "n_successful_rollouts": 49,
  "n_failed_rollouts": 1,
  "failure_rate": 0.02,
  "avg_trajectory_length": 47.3,
  "reward_summary": { "mean": 12.3, "std": 1.1, "min": 10.0, "max": 14.0 },
  "ego_reward_summary": { "mean": 4.5, "std": 0.7, "min": 3.2, "max": 5.4 },
  "teammate_reward_summary": { "mean": 7.8, "std": 0.9, "min": 6.4, "max": 8.9 },
  "per_rollout_reward": [12.0, null, 14.0],
  "per_rollout_ego_reward": [4.0, null, 5.0],
  "per_rollout_teammate_reward": [8.0, null, 9.0],
  "rollout_errors": [null, "Traceback ...", null]
}
```

| Key | Present in | Description |
|-----|-----------|-------------|
| `idx` | all | Zero-based teammate index |
| `name` | scripted evals | Scripted policy name |
| `seed_dir` | self-play evals | Absolute path to the self-play seed directory |
| `policy_path` | policy-code evals | Absolute path to the selected policy artifact |
| `n_rollouts` | all | Total rollouts attempted for this teammate |
| `n_successful_rollouts` | all | Rollouts that completed without error |
| `n_failed_rollouts` | all | Rollouts that raised an exception |
| `failure_rate` | all | `n_failed_rollouts / n_rollouts` |
| `avg_trajectory_length` | all | Mean steps across successful rollouts |
| `reward_summary` | all | Team-total reward stats for this teammate |
| `ego_reward_summary` | all | Ego-agent reward stats for this teammate |
| `teammate_reward_summary` | all | Summed teammate reward stats for this teammate |
| `per_rollout_reward` | all | Team-total reward per rollout; `null` for failed rollouts |
| `per_rollout_ego_reward` | all | Ego-agent reward per rollout; `null` for failed rollouts |
| `per_rollout_teammate_reward` | all | Summed teammate reward per rollout; `null` for failed rollouts |
| `rollout_errors` | all | Error traceback string per rollout; `null` for successful ones |

Reward semantics:

- `per_rollout_reward` is the total reward over all agents in `ego_team`.
- `per_rollout_ego_reward` is the reward collected by `ego_team.ego_agent`.
- `per_rollout_teammate_reward` is the sum of rewards collected by all `ego_team.teammate_*` agents.

---

## Notes For New Evaluators

To produce compatible output:

1. Collect per-rollout team, ego, and teammate rewards. Failed rollouts should record `null` for each reward channel.
2. Build teammate entries with the schema above.
3. Compute aggregate summaries by flattening non-null values across all teammates.
4. Write the top-level dict to `evaluation/<filename>.json`.

The helpers `_build_rollout_summary`, `_build_teammate_eval_summary`, and `_save_summary` in `popgen/cooperator/evaluation.py` already implement the schema used by `CooperatorEvaluator` and `BaselineEvaluator`.
