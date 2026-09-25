import gc
import importlib
import json
import logging
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from pqdm.processes import pqdm
import torch
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from popgen.agents.router import MultiPolicyRouter
from popgen.cooperator.util import load_ego_policy_and_env
from popgen.utils.logging import save_rollout

from popgen.cooperator.selfplay_util import (
    SelfPlayPolicyTensorAdapter as _SelfPlayPolicyTensorAdapter,
    close_env_safely,
    load_selfplay_policy as _load_single_self_play_policy,
)


def _extract_ego_team_reward(states: list[dict[str, Any]]) -> float | None:
    """Sum ego_team rewards across all steps and agents."""
    total = 0.0
    found_any = False
    for step in states:
        for agent in step.get("ego_team", {}).values():
            reward = agent.get("reward")
            if reward is not None:
                total += float(reward)
                found_any = True
    return total if found_any else None


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


def _stringify_rollout_error(error: Any) -> str:
    if isinstance(error, BaseException):
        detail = str(error).strip() or "<no message>"
        formatted_tb = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
        if formatted_tb:
            return f"{type(error).__name__}: {detail}\n{formatted_tb}"
        return f"{type(error).__name__}: {detail}"
    return str(error)


def _build_rollout_summary(
    n_eval_rollouts: int,
    rollout_errors: list[str | None],
    trajectory_lengths: list[int],
    rollout_paths: list[str | None],
    per_rollout_ego_team_rewards: list[float | None],
    rollout_seeds: list[int],
    base_seed: int,
    elapsed: float,
) -> dict[str, Any]:
    del rollout_paths, rollout_seeds, base_seed, elapsed
    n_failed = sum(1 for err in rollout_errors if err is not None)
    avg_traj_len = (sum(trajectory_lengths) / len(trajectory_lengths)) if trajectory_lengths else 0.0
    reward_values = [reward for reward in per_rollout_ego_team_rewards if reward is not None]
    return {
        "n_rollouts": n_eval_rollouts,
        "n_successful_rollouts": n_eval_rollouts - n_failed,
        "n_failed_rollouts": n_failed,
        "failure_rate": n_failed / n_eval_rollouts,
        "avg_trajectory_length": avg_traj_len,
        "reward_summary": _build_scalar_summary(reward_values),
        "per_rollout_reward": per_rollout_ego_team_rewards,
        "rollout_errors": rollout_errors,
    }


def _count_rollout_outcomes(entries: list[dict[str, Any]]) -> tuple[int, int, int]:
    total = sum(int(entry["n_rollouts"]) for entry in entries)
    successful = sum(int(entry["n_successful_rollouts"]) for entry in entries)
    failed = sum(int(entry["n_failed_rollouts"]) for entry in entries)
    return total, successful, failed


def _normalize_evaluation_set_name(evaluation_set: str) -> str:
    return str(evaluation_set).strip().lower().replace("-", "_")


def _policies_from_behavior_set(behavior_set: Any) -> list[Callable[[Any], Any]]:
    if not isinstance(behavior_set, list):
        return []
    scripted_policies = [entry.get("policy_cls") for entry in behavior_set if isinstance(entry, dict)]
    return [policy for policy in scripted_policies if policy is not None]


def _resolve_evaluation_scripted_policies(
    env_module,
    evaluation_set: str = "standard",
) -> list[Callable[[Any], Any]]:
    normalized = _normalize_evaluation_set_name(evaluation_set)

    if hasattr(env_module, "get_named_evaluation_scripted_policies"):
        scripted_policies = env_module.get_named_evaluation_scripted_policies(normalized)
        if not isinstance(scripted_policies, list) or not scripted_policies:
            raise ValueError(
                "env_module.get_named_evaluation_scripted_policies() must return a non-empty list."
            )
        return scripted_policies

    if normalized in {"standard", "default"} and hasattr(env_module, "get_evaluation_scripted_policies"):
        scripted_policies = env_module.get_evaluation_scripted_policies()
        if not isinstance(scripted_policies, list) or not scripted_policies:
            raise ValueError(
                "env_module.get_evaluation_scripted_policies() must return a non-empty list."
            )
        return scripted_policies

    behavior_set_attr = (
        "EVALUATION_BEHAVIOR_SET"
        if normalized in {"standard", "default"}
        else f"{normalized.upper()}_EVALUATION_BEHAVIOR_SET"
    )
    if hasattr(env_module, behavior_set_attr):
        scripted_policies = _policies_from_behavior_set(getattr(env_module, behavior_set_attr))
        if scripted_policies:
            return scripted_policies

    if normalized in {"standard", "default"} and hasattr(env_module, "ScriptedPolicy"):
        return [env_module.ScriptedPolicy]

    raise AttributeError(
        f"env_module does not expose scripted policies for evaluation_set={evaluation_set!r}."
    )


def _build_teammate_evaluation_summary(
    n_eval_rollouts: int,
    teammates: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rewards = [
        reward
        for teammate in teammates
        for reward in teammate["per_rollout_reward"]
        if reward is not None
    ]
    return {
        "n_teammates": len(teammates),
        "n_eval_rollouts": n_eval_rollouts,
        "aggregate_reward_summary": _build_scalar_summary(all_rewards),
        "teammates": teammates,
        "checkpoint_source": "latest",
    }


def _build_rollout_error_result(rollout_idx: int, error: Any) -> dict[str, Any]:
    return {
        "rollout_idx": rollout_idx,
        "rollout_error": None if error is None else _stringify_rollout_error(error),
        "trajectory_length": 0,
        "rollout_path": None,
        "ego_team_reward": None,
    }


def _build_fresh_rollout_env(env_module, reloaded_env=None):
    fresh_env = env_module.create_env(num_envs=1, seed=0, device="cpu")
    if reloaded_env is not None and isinstance(reloaded_env, TransformedEnv):
        fresh_env = TransformedEnv(fresh_env, reloaded_env.transform.clone())
    return fresh_env


def _build_rollout_policy(eval_kind: str, args: dict[str, Any], env_module) -> tuple[Any, Any]:
    reload_context = getattr(env_module, "get_cooperator_reload_context", None)
    evaluation_set = _normalize_evaluation_set_name(args.get("evaluation_set", "standard"))

    if eval_kind == "cooperator_scripted":
        reload_ctx = reload_context() if callable(reload_context) else nullcontext()
        with reload_ctx:
            ego_policy, reloaded_env = load_ego_policy_and_env(
                args["cooperator_dir"],
                env_module.create_env,
            )
        ego_env = _build_fresh_rollout_env(env_module, reloaded_env=reloaded_env)
        close_env_safely(reloaded_env)
        ego_policy = ego_policy.to(getattr(ego_env, "device", "cpu"))
        scripted_policy_factory = _resolve_evaluation_scripted_policies(
            env_module,
            evaluation_set=evaluation_set,
        )[args["scripted_policy_idx"]]
        temp_env = env_module.create_env(num_envs=1, seed=0, device="cpu")
        try:
            scripted_policy = scripted_policy_factory(temp_env)
        finally:
            close_env_safely(temp_env)
        return MultiPolicyRouter({"ego": ego_policy, "teammate": [scripted_policy]}), ego_env

    if eval_kind == "cooperator_vs_population":
        reload_ctx = reload_context() if callable(reload_context) else nullcontext()
        with reload_ctx:
            ego_policy, reloaded_env = load_ego_policy_and_env(
                args["cooperator_dir"],
                env_module.create_env,
            )
        ego_env = _build_fresh_rollout_env(env_module, reloaded_env=reloaded_env)
        close_env_safely(reloaded_env)
        ego_policy = ego_policy.to(getattr(ego_env, "device", "cpu"))
        teammate_policy = _load_single_self_play_policy(
            Path(args["teammate_seed_dir"]),
            allow_dynamic_stubs=True,
            move_policy_to_cpu=True,
            reload_context=reload_context,
        )
        adapted_teammate = _SelfPlayPolicyTensorAdapter(teammate_policy)
        return MultiPolicyRouter({"ego": ego_policy, "teammate": [adapted_teammate]}), ego_env

    ego_env = env_module.create_env(num_envs=1, seed=0, device="cpu")
    ego_policy = _load_single_self_play_policy(
        Path(args["ego_seed_dir"]),
        allow_dynamic_stubs=True,
        move_policy_to_cpu=True,
        reload_context=reload_context,
    )
    adapted_ego = _SelfPlayPolicyTensorAdapter(ego_policy)

    if eval_kind == "self_play_self_pair":
        return MultiPolicyRouter({"ego": [adapted_ego], "teammate": [adapted_ego]}), ego_env

    if eval_kind == "self_play_vs_population":
        teammate_policy = _load_single_self_play_policy(
            Path(args["teammate_seed_dir"]),
            allow_dynamic_stubs=True,
            move_policy_to_cpu=True,
            reload_context=reload_context,
        )
        adapted_teammate = _SelfPlayPolicyTensorAdapter(teammate_policy)
        return MultiPolicyRouter({"ego": [adapted_ego], "teammate": [adapted_teammate]}), ego_env

    if eval_kind == "self_play_vs_scripted":
        scripted_policy_factory = _resolve_evaluation_scripted_policies(
            env_module,
            evaluation_set=evaluation_set,
        )[args["scripted_policy_idx"]]
        temp_env = env_module.create_env(num_envs=1, seed=0, device="cpu")
        try:
            scripted_policy = scripted_policy_factory(temp_env)
        finally:
            close_env_safely(temp_env)
        return MultiPolicyRouter({"ego": [adapted_ego], "teammate": [scripted_policy]}), ego_env

    raise ValueError(f"Unsupported eval kind: {eval_kind}")


def _run_eval_rollout_worker(args: dict[str, Any]) -> dict[str, Any]:
    env_module = importlib.import_module(args["env_module_name"])
    ego_env = None
    rollout_policy = None
    states = None
    images = None
    try:
        rollout_policy, ego_env = _build_rollout_policy(args["eval_kind"], args, env_module)
        rollout_context = (
            set_exploration_type(ExplorationType.DETERMINISTIC)
            if args.get("deterministic_actions", False)
            else nullcontext()
        )
        with rollout_context:
            states, images, error = env_module.rollout_fn(
                rollout_policy,
                env=ego_env,
                seed=args["rollout_seed"],
            )
        ego_team_reward = _extract_ego_team_reward(states) if error is None and states else None

        if error is None and states is not None:
            save_rollout(
                rollout_data=states,
                images=images or [],
                save_path=args["rollout_path"],
            )

        return {
            "rollout_idx": args["rollout_idx"],
            "rollout_error": None if error is None else _stringify_rollout_error(error),
            "trajectory_length": len(states) if error is None and states is not None else 0,
            "rollout_path": args["rollout_path"] if error is None and states is not None else None,
            "ego_team_reward": ego_team_reward,
        }
    except Exception as exc:
        return _build_rollout_error_result(args["rollout_idx"], exc)
    finally:
        close_env_safely(ego_env)
        ego_env = None
        rollout_policy = None
        states = None
        images = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_rollouts(
    n_eval_rollouts: int,
    env_module,
    eval_kind: str,
    rollout_path_fn: Callable[[int], Path],
    num_workers: int,
    seed: int | None = None,
    deterministic_actions: bool = False,
    cooperator_dir: str | Path | None = None,
    ego_seed_dir: str | Path | None = None,
    teammate_seed_dir: str | Path | None = None,
    scripted_policy_idx: int | None = None,
    evaluation_set: str = "standard",
) -> tuple[list[str | None], list[int], list[str | None], list[float | None], list[int], int]:
    base_seed = 0 if seed is None else int(seed)
    rollout_seeds = [base_seed + rollout_idx * 1000 for rollout_idx in range(n_eval_rollouts)]
    env_module_name = getattr(env_module, "__name__", None)
    if env_module_name is None:
        raise TypeError("run_rollouts requires an importable env module.")

    rollout_args = []
    for rollout_idx, rollout_seed in enumerate(rollout_seeds):
        rollout_args.append(
            {
                "rollout_idx": rollout_idx,
                "rollout_seed": rollout_seed,
                "rollout_path": str(rollout_path_fn(rollout_idx)),
                "env_module_name": env_module_name,
                "eval_kind": eval_kind,
                "deterministic_actions": bool(deterministic_actions),
                "cooperator_dir": None if cooperator_dir is None else str(cooperator_dir),
                "ego_seed_dir": None if ego_seed_dir is None else str(ego_seed_dir),
                "teammate_seed_dir": None if teammate_seed_dir is None else str(teammate_seed_dir),
                "scripted_policy_idx": scripted_policy_idx,
                "evaluation_set": _normalize_evaluation_set_name(evaluation_set),
            }
        )

    rollout_results = pqdm(
        rollout_args,
        _run_eval_rollout_worker,
        n_jobs=max(1, int(num_workers)),
        desc="Running self-play eval rollouts",
    )

    normalized_rollout_results: list[dict[str, Any]] = []
    for rollout_idx, result in enumerate(rollout_results):
        if isinstance(result, BaseException):
            normalized_rollout_results.append(_build_rollout_error_result(rollout_idx, result))
        elif isinstance(result, dict):
            normalized_rollout_results.append(result)
        else:
            normalized_rollout_results.append(
                _build_rollout_error_result(
                    rollout_idx,
                    TypeError(f"Unexpected pqdm rollout result type: {type(result).__name__}"),
                )
            )

    rollout_errors = [result["rollout_error"] for result in normalized_rollout_results]
    trajectory_lengths = [
        result["trajectory_length"]
        for result in normalized_rollout_results
        if result["rollout_error"] is None
    ]
    rollout_paths = [result["rollout_path"] for result in normalized_rollout_results]
    per_rollout_ego_team_rewards = [result["ego_team_reward"] for result in normalized_rollout_results]

    return (
        rollout_errors,
        trajectory_lengths,
        rollout_paths,
        per_rollout_ego_team_rewards,
        rollout_seeds,
        base_seed,
    )


def _write_summary(output_dir: Path, file_name: str, summary: dict) -> dict:
    summary_path = output_dir / file_name
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
