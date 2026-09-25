import logging
import os
import re
import sys
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

from eval_common import (
    _build_rollout_summary,
    _count_rollout_outcomes,
    _normalize_evaluation_set_name,
    _resolve_evaluation_scripted_policies,
    _run_rollouts,
    _write_summary,
    _build_scalar_summary,
)
from pipeline_utils import _enumerate_seed_dirs, resolve_env_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def _seed_dir_id(seed_dir: str | Path) -> int:
    match = _SEED_DIR_RE.fullmatch(Path(seed_dir).name)
    if match is None:
        raise ValueError(f"Expected a seed directory named seed_<id>, got: {seed_dir}")
    return int(match.group(1))


def _resolve_evaluation_scope(
    self_play_results_dir: str | Path,
) -> tuple[Path, list[Path], list[Path], bool]:
    input_path = Path(self_play_results_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"self_play_results_dir does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"self_play_results_dir must be a directory: {input_path}")

    if _SEED_DIR_RE.fullmatch(input_path.name):
        population_path = input_path.parent.resolve()
        population_seed_dirs = _enumerate_seed_dirs(population_path)
        target_seed_dirs = [seed_dir for seed_dir in population_seed_dirs if seed_dir.resolve() == input_path]
        if not target_seed_dirs:
            raise FileNotFoundError(f"Could not find target seed directory in parent run: {input_path}")
        return input_path, target_seed_dirs, population_seed_dirs, True

    population_seed_dirs = _enumerate_seed_dirs(input_path)
    return input_path, population_seed_dirs, population_seed_dirs, False


def evaluate_self_play_vs_population(
    ego_idx: int,
    ego_seed_dir: str | Path,
    teammate_seed_dirs: list[str | Path],
    env_module,
    n_eval_rollouts: int,
    eval_dir: Path,
    num_workers: int,
    seed: int | None = None,
    seed_name: str | None = None,
    deterministic_actions: bool = True,
) -> dict:
    _seed_label = seed_name if seed_name is not None else f"seed_{ego_idx}"
    indexed_teammate_dirs = [
        (_seed_dir_id(seed_dir), Path(seed_dir))
        for seed_dir in teammate_seed_dirs
        if Path(seed_dir).resolve() != Path(ego_seed_dir).resolve()
    ]
    if not indexed_teammate_dirs:
        raise ValueError(f"Cannot evaluate seed {ego_idx}: no other policies in the population.")

    eval_dir.mkdir(parents=True, exist_ok=True)

    started_at = time()
    per_teammate_summaries: list[dict[str, Any]] = []
    for teammate_idx, teammate_seed_dir in indexed_teammate_dirs:
        rollout_started_at = time()
        (
            rollout_errors,
            trajectory_lengths,
            rollout_paths,
            per_rollout_ego_team_rewards,
            rollout_seeds,
            base_seed,
        ) = _run_rollouts(
            n_eval_rollouts=n_eval_rollouts,
            env_module=env_module,
            eval_kind="self_play_vs_population",
            rollout_path_fn=lambda i, idx=teammate_idx: eval_dir / f"{_seed_label}_rollout_{i}_vs_seed_{idx}",
            num_workers=num_workers,
            seed=seed,
            deterministic_actions=deterministic_actions,
            ego_seed_dir=ego_seed_dir,
            teammate_seed_dir=teammate_seed_dir,
        )
        teammate_summary = _build_rollout_summary(
            n_eval_rollouts,
            rollout_errors,
            trajectory_lengths,
            rollout_paths,
            per_rollout_ego_team_rewards,
            rollout_seeds,
            base_seed,
            time() - rollout_started_at,
        )
        teammate_summary["teammate_idx"] = teammate_idx
        per_teammate_summaries.append(teammate_summary)

    all_rewards = [
        reward
        for teammate_summary in per_teammate_summaries
        for reward in teammate_summary["per_rollout_ego_team_reward"]
        if reward is not None
    ]

    summary = {
        "ego_seed_idx": ego_idx,
        "n_teammates": len(indexed_teammate_dirs),
        "teammate_indices": [idx for idx, _ in indexed_teammate_dirs],
        "n_eval_rollouts_per_teammate": n_eval_rollouts,
        "aggregate_ego_team_reward_summary": _build_scalar_summary(all_rewards),
        "per_teammate_summaries": per_teammate_summaries,
        "sampling_strategy": "each_population_teammate_excluding_self",
        "seed": seed,
        "base_seed": 0 if seed is None else int(seed),
        "elapsed_sec": time() - started_at,
    }
    return _write_summary(eval_dir, f"{_seed_label}_vs_population_eval.json", summary)


def evaluate_self_play_with_self(
    ego_idx: int,
    ego_seed_dir: str | Path,
    env_module,
    n_eval_rollouts: int,
    eval_dir: Path,
    num_workers: int,
    seed: int | None = None,
    seed_name: str | None = None,
    deterministic_actions: bool = True,
) -> dict:
    _seed_label = seed_name if seed_name is not None else f"seed_{ego_idx}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    started_at = time()
    (
        rollout_errors,
        trajectory_lengths,
        rollout_paths,
        per_rollout_ego_team_rewards,
        rollout_seeds,
        base_seed,
    ) = _run_rollouts(
        n_eval_rollouts=n_eval_rollouts,
        env_module=env_module,
        eval_kind="self_play_self_pair",
        rollout_path_fn=lambda i: eval_dir / f"{_seed_label}_rollout_{i}_vs_self",
        num_workers=num_workers,
        seed=seed,
        deterministic_actions=deterministic_actions,
        ego_seed_dir=ego_seed_dir,
    )

    summary = _build_rollout_summary(
        n_eval_rollouts,
        rollout_errors,
        trajectory_lengths,
        rollout_paths,
        per_rollout_ego_team_rewards,
        rollout_seeds,
        base_seed,
        time() - started_at,
    )
    summary.update(
        {
            "ego_seed_idx": ego_idx,
            "sampling_strategy": "self_pair",
            "seed": seed,
        }
    )
    return _write_summary(eval_dir, f"{_seed_label}_vs_self_eval.json", summary)


def evaluate_self_play_vs_scripted(
    ego_idx: int,
    ego_seed_dir: str | Path,
    env_module,
    n_eval_rollouts: int,
    eval_dir: Path,
    num_workers: int,
    seed: int | None = None,
    seed_name: str | None = None,
    deterministic_actions: bool = True,
    evaluation_set: str = "standard",
    output_file_name: str | None = None,
) -> dict:
    _seed_label = seed_name if seed_name is not None else f"seed_{ego_idx}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    normalized_set = _normalize_evaluation_set_name(evaluation_set)
    scripted_policy_factories = _resolve_evaluation_scripted_policies(
        env_module,
        evaluation_set=normalized_set,
    )

    started_at = time()
    per_scripted_policy_summaries: list[dict[str, Any]] = []
    scripted_policy_names: list[str] = []
    for policy_idx, scripted_policy_factory in enumerate(scripted_policy_factories):
        policy_name = getattr(scripted_policy_factory, "__name__", f"scripted_policy_{policy_idx}")
        scripted_policy_names.append(str(policy_name))
        rollout_started_at = time()

        (
            rollout_errors,
            trajectory_lengths,
            rollout_paths,
            per_rollout_ego_team_rewards,
            rollout_seeds,
            base_seed,
        ) = _run_rollouts(
            n_eval_rollouts=n_eval_rollouts,
            env_module=env_module,
            eval_kind="self_play_vs_scripted",
            rollout_path_fn=lambda i, idx=policy_idx: eval_dir / f"{_seed_label}_rollout_{i}_vs_scripted_{idx}",
            num_workers=num_workers,
            seed=seed,
            deterministic_actions=deterministic_actions,
            ego_seed_dir=ego_seed_dir,
            scripted_policy_idx=policy_idx,
            evaluation_set=normalized_set,
        )
        policy_summary = _build_rollout_summary(
            n_eval_rollouts,
            rollout_errors,
            trajectory_lengths,
            rollout_paths,
            per_rollout_ego_team_rewards,
            rollout_seeds,
            base_seed,
            time() - rollout_started_at,
        )
        policy_summary.update(
            {
                "sampling_strategy": "scripted_policy",
                "evaluation_set": normalized_set,
                "scripted_policy_idx": policy_idx,
                "scripted_policy_name": str(policy_name),
            }
        )
        per_scripted_policy_summaries.append(policy_summary)

    all_rewards = [
        reward
        for policy_summary in per_scripted_policy_summaries
        for reward in policy_summary["per_rollout_ego_team_reward"]
        if reward is not None
    ]

    summary = {
        "ego_seed_idx": ego_idx,
        "n_scripted_policies": len(scripted_policy_factories),
        "scripted_policy_names": scripted_policy_names,
        "n_eval_rollouts_per_policy": n_eval_rollouts,
        "aggregate_ego_team_reward_summary": _build_scalar_summary(all_rewards),
        "per_scripted_policy_summaries": per_scripted_policy_summaries,
        "evaluation_set": normalized_set,
        "base_seed": 0 if seed is None else int(seed),
        "elapsed_sec": time() - started_at,
    }
    summary_file_name = output_file_name
    if summary_file_name is None:
        summary_file_name = f"{_seed_label}_vs_scripted_eval.json"
        if normalized_set not in {"standard", "default"}:
            summary_file_name = f"{_seed_label}_vs_{normalized_set}_scripted_eval.json"
    return _write_summary(eval_dir, summary_file_name, summary)


def run_self_play_population_evaluation(
    self_play_results_dir: str | Path,
    env_name: str,
    n_eval_rollouts: int,
    num_workers: int,
    seed: int | None = None,
) -> Path:
    import torch

    env_module = resolve_env_module(env_name, required_attrs=("rollout_fn", "create_env"))
    results_path, target_seed_dirs, population_seed_dirs, single_seed_mode = _resolve_evaluation_scope(
        self_play_results_dir
    )
    n_population_policies = len(population_seed_dirs)

    logging.info("=" * 60)
    logging.info("ENUMERATING SELF-PLAY POPULATION")
    logging.info("=" * 60)
    logging.info(f"Found {n_population_policies} self-play seed directories")
    if single_seed_mode:
        logging.info("Single-seed mode: evaluating only %s", target_seed_dirs[0].name)
    else:
        logging.info("Population mode: evaluating all seeds under %s", results_path)

    per_seed_self_summaries: list[dict[str, Any]] = []
    per_seed_population_summaries: list[dict[str, Any]] = []
    per_seed_scripted_summaries: list[dict[str, Any]] = []

    for seed_dir in target_seed_dirs:
        seed_dir_name = seed_dir.name
        seed_idx = _seed_dir_id(seed_dir)
        seed_eval_dir = seed_dir / "evaluation"

        # logging.info("=" * 60)
        # logging.info("EVALUATING SEED %s vs SELF (%d rollouts)", seed_dir_name, n_eval_rollouts)
        # logging.info("=" * 60)
        # self_summary = evaluate_self_play_with_self(
        #     ego_idx=seed_idx,
        #     ego_seed_dir=seed_dir,
        #     env_module=env_module,
        #     n_eval_rollouts=n_eval_rollouts,
        #     eval_dir=seed_eval_dir,
        #     num_workers=num_workers,
        #     seed=seed,
        #     seed_name=seed_dir_name,
        # )
        # torch.cuda.empty_cache()
        # per_seed_self_summaries.append(self_summary)
        # logging.info("Rollouts:    %s", self_summary["n_rollouts"])
        # logging.info("Successful:  %s", self_summary["n_successful_rollouts"])
        # logging.info("Failed:      %s", self_summary["n_failed_rollouts"])
        # logging.info("Summary saved: %s", self_summary["summary_path"])

        # if n_population_policies >= 2:
        #     logging.info("=" * 60)
        #     logging.info(
        #         "EVALUATING SEED %s vs POPULATION (%d rollouts per teammate)",
        #         seed_dir_name,
        #         n_eval_rollouts,
        #     )
        #     logging.info("=" * 60)
        #     population_summary = evaluate_self_play_vs_population(
        #         ego_idx=seed_idx,
        #         ego_seed_dir=seed_dir,
        #         teammate_seed_dirs=population_seed_dirs,
        #         env_module=env_module,
        #         n_eval_rollouts=n_eval_rollouts,
        #         eval_dir=seed_eval_dir,
        #         num_workers=num_workers,
        #         seed=seed,
        #         seed_name=seed_dir_name,
        #     )
        #     torch.cuda.empty_cache()
        #     per_seed_population_summaries.append(population_summary)
        #     total_rollouts, successful_rollouts, failed_rollouts = _count_rollout_outcomes(
        #         population_summary["per_teammate_summaries"]
        #     )
        #     logging.info("Teammates:   %s", population_summary["n_teammates"])
        #     logging.info("Rollouts:    %s", total_rollouts)
        #     logging.info("Successful:  %s", successful_rollouts)
        #     logging.info("Failed:      %s", failed_rollouts)
        #     logging.info("Summary saved: %s", population_summary["summary_path"])
        # else:
        #     logging.info("Skipping population evaluation for %s: only one seed available.", seed_dir_name)

    try:
        _resolve_evaluation_scripted_policies(env_module)
    except AttributeError:
        has_scripted = False
        logging.info("Skipping scripted evaluation (env module defines no scripted evaluation set).")
    else:
        has_scripted = True
        for seed_dir in target_seed_dirs:
            seed_dir_name = seed_dir.name
            seed_idx = _seed_dir_id(seed_dir)
            seed_eval_dir = seed_dir / "evaluation"
            logging.info("=" * 60)
            logging.info(f"EVALUATING SEED {seed_dir_name} vs SCRIPTED ({n_eval_rollouts} rollouts per policy)")
            logging.info("=" * 60)
            summary = evaluate_self_play_vs_scripted(
                ego_idx=seed_idx,
                ego_seed_dir=seed_dir,
                env_module=env_module,
                n_eval_rollouts=n_eval_rollouts,
                eval_dir=seed_eval_dir,
                num_workers=num_workers,
                seed=seed,
                seed_name=seed_dir_name,
            )
            torch.cuda.empty_cache()
            per_seed_scripted_summaries.append(summary)
            total_rollouts, successful_rollouts, failed_rollouts = _count_rollout_outcomes(
                summary["per_scripted_policy_summaries"]
            )
            logging.info(f"Policies:    {summary['n_scripted_policies']}")
            logging.info(f"Rollouts:    {total_rollouts}")
            logging.info(f"Successful:  {successful_rollouts}")
            logging.info(f"Failed:      {failed_rollouts}")
            logging.info(f"Summary saved: {summary['summary_path']}")

    all_scripted_rewards: list[float] = [
        reward
        for summary in per_seed_scripted_summaries
        for policy_summary in summary["per_scripted_policy_summaries"]
        for reward in policy_summary["per_rollout_ego_team_reward"]
        if reward is not None
    ]
    all_self_rewards: list[float] = [
        reward
        for summary in per_seed_self_summaries
        for reward in summary["per_rollout_ego_team_reward"]
        if reward is not None
    ]
    all_population_rewards: list[float] = [
        reward
        for summary in per_seed_population_summaries
        for teammate_summary in summary["per_teammate_summaries"]
        for reward in teammate_summary["per_rollout_ego_team_reward"]
        if reward is not None
    ]

    if not single_seed_mode:
        aggregate_summary: dict[str, Any] = {
            "n_seeds": n_population_policies,
            "n_eval_rollouts_per_teammate": n_eval_rollouts,
            "n_eval_rollouts_per_self_pair": n_eval_rollouts,
            "aggregate_vs_self_ego_team_reward_summary": _build_scalar_summary(all_self_rewards),
            "per_seed_vs_self_summaries": per_seed_self_summaries,
        }
        if per_seed_population_summaries:
            aggregate_summary["aggregate_vs_population_ego_team_reward_summary"] = _build_scalar_summary(
                all_population_rewards
            )
            aggregate_summary["per_seed_vs_population_summaries"] = per_seed_population_summaries
        if has_scripted:
            aggregate_summary["n_eval_rollouts_per_scripted_policy"] = n_eval_rollouts
            aggregate_summary["aggregate_vs_scripted_ego_team_reward_summary"] = _build_scalar_summary(
                all_scripted_rewards
            )
            aggregate_summary["per_seed_vs_scripted_summaries"] = per_seed_scripted_summaries

        _write_summary(results_path, "population_eval.json", aggregate_summary)

    logging.info("=" * 60)
    logging.info("DONE")
    logging.info(f"Self-play population evaluation results: {results_path}")
    logging.info("=" * 60)
    return results_path


def resolve_self_play_results_dir(cfg: DictConfig) -> str:
    if cfg.cooperator.self_play_results_dir is not None:
        return to_absolute_path(str(cfg.cooperator.self_play_results_dir))
    return to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    """Evaluate a self-play population against itself (cross-play) and against scripted teammates.

    Run from this directory:
        python self_play_population_evaluation.py env=lbf
        python self_play_population_evaluation.py env=lbf cooperator.self_play_results_dir=/path/to/sp_run

    Key config fields (cfg/config.yaml):
        env                               Environment name (lbf, football, smacv2, ...)
        cooperator.self_play_results_dir  Path to the self-play run directory (seed_*/ subdirs).
                                          If unset, uses <self_play.output_dir>/<self_play.run_name>.
        evaluation.n_eval_rollouts        Number of rollouts per evaluation pair (default: 50)
        evaluation.num_workers            Parallel rollout workers (default: 8)
        evaluation.seed                   Base random seed for rollouts (default: null)
    """
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_self_play_population_evaluation(
        self_play_results_dir=resolve_self_play_results_dir(cfg),
        env_name=cfg.env,
        n_eval_rollouts=cfg.evaluation.n_eval_rollouts,
        num_workers=cfg.evaluation.num_workers,
        seed=cfg.evaluation.seed,
    )


if __name__ == "__main__":
    main()
