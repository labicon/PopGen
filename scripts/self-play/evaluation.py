import inspect
import json
import logging
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# Ensure project root is on path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from popgen.utils.logging import save_rollout


# ---------------------------------------------------------------------------
# Rollout execution helpers (inlined from popgen.cooperator.evaluation)
# ---------------------------------------------------------------------------


def _rollout_supports_return_info(rollout_fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(rollout_fn)
    except (TypeError, ValueError):
        return False

    if "return_info" in signature.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _unpack_rollout_result(
    rollout_result: Any,
) -> tuple[Any, Any, Any, dict[str, Any] | None]:
    if not isinstance(rollout_result, (tuple, list)):
        raise TypeError(
            f"rollout_fn must return a tuple/list, got {type(rollout_result).__name__}."
        )

    if len(rollout_result) < 3:
        raise ValueError(
            f"rollout_fn must return at least 3 elements, got {len(rollout_result)}."
        )

    states = rollout_result[0]
    images = rollout_result[1]
    error = rollout_result[2]
    rollout_info = rollout_result[3] if len(rollout_result) >= 4 else None
    if rollout_info is not None and not isinstance(rollout_info, dict):
        rollout_info = None
    return states, images, error, rollout_info


def _coerce_reward_totals_by_group(
    rollout_info: dict[str, Any] | None,
) -> dict[str, float] | None:
    if not isinstance(rollout_info, dict):
        return None

    reward_totals_raw = rollout_info.get("reward_totals_by_group")
    if not isinstance(reward_totals_raw, dict):
        return None

    reward_totals: dict[str, float] = {}
    for group, value in reward_totals_raw.items():
        try:
            reward_totals[str(group)] = float(value)
        except (TypeError, ValueError):
            continue
    return reward_totals or None


def _coerce_reward_total_all_groups(
    rollout_info: dict[str, Any] | None,
    reward_totals_by_group: dict[str, float] | None,
) -> float | None:
    if isinstance(rollout_info, dict):
        try:
            reward_total_raw = rollout_info.get("reward_total_all_groups")
            if reward_total_raw is not None:
                return float(reward_total_raw)
        except (TypeError, ValueError):
            pass

    if reward_totals_by_group:
        return float(sum(reward_totals_by_group.values()))
    return None


def _build_scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        "mean": mean_value,
        "std": variance**0.5,
        "min": min(values),
        "max": max(values),
    }


def summarize_rollout_rewards(
    rollout_errors: list[str | None],
    per_rollout_reward_totals_by_group: list[dict[str, float] | None],
    per_rollout_reward_total_all_groups: list[float | None],
) -> dict[str, Any]:
    """Aggregate per-rollout reward data into summary statistics.

    Only successful rollouts (error is None) contribute to the statistics.
    """
    group_totals_across_rollouts: dict[str, list[float]] = {}
    total_all_groups_values: list[float] = []
    n_rollouts_with_reward = 0

    for rollout_idx, error in enumerate(rollout_errors):
        if error is not None:
            continue

        reward_totals_by_group = per_rollout_reward_totals_by_group[rollout_idx]
        reward_total_all_groups = per_rollout_reward_total_all_groups[rollout_idx]

        if reward_totals_by_group:
            for group, value in reward_totals_by_group.items():
                group_totals_across_rollouts.setdefault(group, []).append(float(value))

        if reward_total_all_groups is None and reward_totals_by_group:
            reward_total_all_groups = float(sum(reward_totals_by_group.values()))

        if reward_total_all_groups is not None:
            total_all_groups_values.append(float(reward_total_all_groups))

        if reward_totals_by_group or reward_total_all_groups is not None:
            n_rollouts_with_reward += 1

    reward_summary_by_group = {
        group: _build_scalar_summary(values)
        for group, values in sorted(group_totals_across_rollouts.items())
    }
    reward_summary_all_groups = _build_scalar_summary(total_all_groups_values)

    return {
        "n_rollouts_with_reward": n_rollouts_with_reward,
        "reward_summary_by_group": reward_summary_by_group,
        "reward_summary_all_groups": reward_summary_all_groups,
    }
from pipeline_utils import (
    SelfPlayPolicyWrapper,
    close_env_safely,
    find_cooperator_dir,
    load_latest_policy_and_env,
    load_self_play_policies,
    resolve_env_module,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def _write_summary(eval_dir: Path, file_name: str, summary: dict) -> dict:
    summary_path = eval_dir / file_name
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def evaluate_random_teammate(
    ego_policy,
    ego_env,
    teammate_policies: list[SelfPlayPolicyWrapper],
    env_module,
    n_eval_rollouts: int,
    eval_dir: Path,
    seed: int | None = None,
) -> dict:
    rng = random.Random(seed)
    sampled_indices: list[int] = []
    rollout_errors: list[str | None] = []
    trajectory_lengths: list[int] = []
    rollout_paths: list[str | None] = []
    per_rollout_reward_totals_by_group: list[dict[str, float] | None] = []
    per_rollout_reward_total_all_groups: list[float | None] = []
    eval_dir.mkdir(parents=True, exist_ok=True)
    supports_return_info = _rollout_supports_return_info(env_module.rollout_fn)

    started_at = time()
    for rollout_idx in range(n_eval_rollouts):
        teammate_idx = rng.randrange(len(teammate_policies))
        sampled_indices.append(teammate_idx)
        teammate = teammate_policies[teammate_idx]

        call_kwargs = {"env": ego_env}
        if supports_return_info:
            call_kwargs["return_info"] = True
        states, images, error, rollout_info = _unpack_rollout_result(
            env_module.rollout_fn([ego_policy, teammate], **call_kwargs)
        )

        reward_totals_by_group = _coerce_reward_totals_by_group(rollout_info)
        reward_total_all_groups = _coerce_reward_total_all_groups(rollout_info, reward_totals_by_group)
        per_rollout_reward_totals_by_group.append(reward_totals_by_group)
        per_rollout_reward_total_all_groups.append(reward_total_all_groups)

        rollout_errors.append(None if error is None else str(error))
        if error is None and states is not None:
            trajectory_lengths.append(len(states))
            rollout_path = eval_dir / f"rollout_{rollout_idx}_with_seed_{teammate_idx}"
            save_rollout(rollout_data=states, images=images or [], save_path=str(rollout_path))
            rollout_paths.append(str(rollout_path))
        else:
            rollout_paths.append(None)

    n_failed = sum(1 for err in rollout_errors if err is not None)
    avg_traj_len = (sum(trajectory_lengths) / len(trajectory_lengths)) if trajectory_lengths else 0.0
    reward_summary = summarize_rollout_rewards(
        rollout_errors=rollout_errors,
        per_rollout_reward_totals_by_group=per_rollout_reward_totals_by_group,
        per_rollout_reward_total_all_groups=per_rollout_reward_total_all_groups,
    )

    histogram: dict[str, int] = {}
    for idx in sampled_indices:
        key = str(idx)
        histogram[key] = histogram.get(key, 0) + 1

    summary = {
        "n_rollouts": n_eval_rollouts,
        "n_successful_rollouts": n_eval_rollouts - n_failed,
        "n_failed_rollouts": n_failed,
        "failure_rate": n_failed / n_eval_rollouts,
        "avg_trajectory_length": avg_traj_len,
        "n_rollouts_with_reward": reward_summary["n_rollouts_with_reward"],
        "per_rollout_reward_totals_by_group": per_rollout_reward_totals_by_group,
        "per_rollout_reward_total_all_groups": per_rollout_reward_total_all_groups,
        "reward_summary_by_group": reward_summary["reward_summary_by_group"],
        "reward_summary_all_groups": reward_summary["reward_summary_all_groups"],
        "sampled_teammate_indices": sampled_indices,
        "teammate_sample_histogram": histogram,
        "rollout_errors": rollout_errors,
        "saved_rollout_paths": rollout_paths,
        "checkpoint_source": "latest",
        "sampling_strategy": "uniform_per_rollout",
        "seed": seed,
        "elapsed_sec": time() - started_at,
    }
    return _write_summary(eval_dir, "random_teammate_eval.json", summary)


def evaluate_scripted_policy(
    ego_policy,
    ego_env,
    env_module,
    n_eval_rollouts: int,
    eval_dir: Path,
) -> dict:
    rollout_errors: list[str | None] = []
    trajectory_lengths: list[int] = []
    rollout_paths: list[str | None] = []
    per_rollout_reward_totals_by_group: list[dict[str, float] | None] = []
    per_rollout_reward_total_all_groups: list[float | None] = []
    eval_dir.mkdir(parents=True, exist_ok=True)
    supports_return_info = _rollout_supports_return_info(env_module.rollout_fn)

    started_at = time()
    for rollout_idx in range(n_eval_rollouts):
        temp_env = env_module.create_env(num_envs=1, seed=0, device="cpu")
        try:
            scripted_policy = env_module.ScriptedPolicy(temp_env)
        finally:
            close_env_safely(temp_env)

        call_kwargs = {"env": ego_env}
        if supports_return_info:
            call_kwargs["return_info"] = True
        states, images, error, rollout_info = _unpack_rollout_result(
            env_module.rollout_fn([ego_policy, scripted_policy], **call_kwargs)
        )

        reward_totals_by_group = _coerce_reward_totals_by_group(rollout_info)
        reward_total_all_groups = _coerce_reward_total_all_groups(rollout_info, reward_totals_by_group)
        per_rollout_reward_totals_by_group.append(reward_totals_by_group)
        per_rollout_reward_total_all_groups.append(reward_total_all_groups)

        rollout_errors.append(None if error is None else str(error))
        if error is None and states is not None:
            trajectory_lengths.append(len(states))
            rollout_path = eval_dir / f"rollout_{rollout_idx}_with_scripted_policy"
            save_rollout(rollout_data=states, images=images or [], save_path=str(rollout_path))
            rollout_paths.append(str(rollout_path))
        else:
            rollout_paths.append(None)

    n_failed = sum(1 for err in rollout_errors if err is not None)
    avg_traj_len = (sum(trajectory_lengths) / len(trajectory_lengths)) if trajectory_lengths else 0.0
    reward_summary = summarize_rollout_rewards(
        rollout_errors=rollout_errors,
        per_rollout_reward_totals_by_group=per_rollout_reward_totals_by_group,
        per_rollout_reward_total_all_groups=per_rollout_reward_total_all_groups,
    )

    summary = {
        "n_rollouts": n_eval_rollouts,
        "n_successful_rollouts": n_eval_rollouts - n_failed,
        "n_failed_rollouts": n_failed,
        "failure_rate": n_failed / n_eval_rollouts,
        "avg_trajectory_length": avg_traj_len,
        "n_rollouts_with_reward": reward_summary["n_rollouts_with_reward"],
        "per_rollout_reward_totals_by_group": per_rollout_reward_totals_by_group,
        "per_rollout_reward_total_all_groups": per_rollout_reward_total_all_groups,
        "reward_summary_by_group": reward_summary["reward_summary_by_group"],
        "reward_summary_all_groups": reward_summary["reward_summary_all_groups"],
        "rollout_errors": rollout_errors,
        "saved_rollout_paths": rollout_paths,
        "checkpoint_source": "latest",
        "sampling_strategy": "scripted_policy",
        "elapsed_sec": time() - started_at,
    }
    return _write_summary(eval_dir, "scripted_policy_eval.json", summary)


def run_evaluation_pipeline(
    run_dir: str | Path,
    env_name: str,
    n_eval_rollouts: int,
    seed: int | None = None,
) -> Path:
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_path}")

    env_module = resolve_env_module(env_name, required_attrs=("rollout_fn",))

    logging.info("=" * 60)
    logging.info("LOADING SELF-PLAY POPULATION")
    logging.info("=" * 60)
    teammate_policies = load_self_play_policies(
        run_path,
        allow_dynamic_stubs=True,
        move_policy_to_cpu=True,
    )
    logging.info(f"Loaded {len(teammate_policies)} self-play policies")

    cooperator_dir = find_cooperator_dir(run_path)

    logging.info("=" * 60)
    logging.info("LOADING EGO (COOPERATOR) POLICY")
    logging.info("=" * 60)
    ego_policy, ego_env, _ = load_latest_policy_and_env(
        cooperator_dir,
        allow_dynamic_stubs=True,
        move_policy_to_cpu=True,
    )

    eval_dir = cooperator_dir / "evaluation"
    try:
        logging.info("=" * 60)
        logging.info("STAGE 3: RANDOM TEAMMATE EVALUATION")
        logging.info("=" * 60)
        random_summary = evaluate_random_teammate(
            ego_policy=ego_policy,
            ego_env=ego_env,
            teammate_policies=teammate_policies,
            env_module=env_module,
            n_eval_rollouts=n_eval_rollouts,
            eval_dir=eval_dir,
            seed=seed,
        )
        logging.info(f"Rollouts:        {random_summary['n_rollouts']}")
        logging.info(f"Successful:      {random_summary['n_successful_rollouts']}")
        logging.info(f"Failed:          {random_summary['n_failed_rollouts']}")
        logging.info(f"Avg traj length: {random_summary['avg_trajectory_length']:.2f}")
        logging.info(f"Summary saved:   {random_summary['summary_path']}")

        if hasattr(env_module, "ScriptedPolicy"):
            if not hasattr(env_module, "create_env"):
                raise AttributeError(f"popgen.envs.{env_name}.utils must define create_env")
            logging.info("=" * 60)
            logging.info("STAGE 4: SCRIPTED POLICY EVALUATION")
            logging.info("=" * 60)
            scripted_summary = evaluate_scripted_policy(
                ego_policy=ego_policy,
                ego_env=ego_env,
                env_module=env_module,
                n_eval_rollouts=n_eval_rollouts,
                eval_dir=eval_dir,
            )
            logging.info(f"Rollouts:        {scripted_summary['n_rollouts']}")
            logging.info(f"Successful:      {scripted_summary['n_successful_rollouts']}")
            logging.info(f"Failed:          {scripted_summary['n_failed_rollouts']}")
            logging.info(f"Avg traj length: {scripted_summary['avg_trajectory_length']:.2f}")
            logging.info(f"Summary saved:   {scripted_summary['summary_path']}")
        else:
            logging.info("Skipping scripted policy evaluation (env_module has no ScriptedPolicy).")
    finally:
        close_env_safely(ego_env)

    logging.info("=" * 60)
    logging.info("DONE")
    logging.info(f"Evaluation results: {eval_dir}")
    logging.info("=" * 60)
    return eval_dir


def resolve_evaluation_run_dir(cfg: DictConfig) -> str:
    if cfg.evaluation.run_dir is not None:
        return to_absolute_path(str(cfg.evaluation.run_dir))
    if cfg.cooperator.self_play_results_dir is not None:
        return to_absolute_path(str(cfg.cooperator.self_play_results_dir))
    return to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_evaluation_pipeline(
        run_dir=resolve_evaluation_run_dir(cfg),
        env_name=cfg.env,
        n_eval_rollouts=cfg.evaluation.n_eval_rollouts,
        seed=cfg.evaluation.seed,
    )


if __name__ == "__main__":
    main()
