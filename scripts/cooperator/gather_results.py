"""Evaluate an existing cooperator checkpoint against scripted and policy-code teammates.

Usage:
    python scripts/cooperator/gather_results.py \
        --cooperator_dir <path/to/cooperator/output/dir> \
        --env point_mass_rendezvous \
        --n_eval_rollouts 50 \
        --results_dir <path/to/population/results>  # optional; enables each-teammate eval
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from popgen.cooperator import CooperatorEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def _log_section(title: str) -> None:
    logging.info("=" * 60)
    logging.info(title)
    logging.info("=" * 60)


def _log_eval_summary(summary: dict) -> None:
    logging.info(f"Rollouts:         {summary['n_rollouts']}")
    logging.info(f"Successful:       {summary['n_successful_rollouts']}")
    logging.info(f"Failed:           {summary['n_failed_rollouts']}")
    logging.info(f"Avg traj length:  {summary['avg_trajectory_length']:.2f}")
    logging.info(f"Summary saved to: {summary['summary_path']}")

def _log_scripted_policies_eval_summary(summary: dict) -> None:
    logging.info(f"Teammates:                   {summary['n_teammates']}")
    logging.info(f"Rollouts per teammate:       {summary['n_eval_rollouts']}")
    agg = summary["aggregate_reward_summary"]
    logging.info(f"Aggregate reward mean/std:   {agg['mean']:.4f} / {agg['std']:.4f}")
    logging.info(f"Summary saved to:            {summary['summary_path']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate an existing cooperator checkpoint.")
    parser.add_argument("--cooperator_dir", required=True, help="Path to cooperator output directory (contains checkpoints/)")
    parser.add_argument("--env", default="point_mass_rendezvous", help="Environment name (default: point_mass_rendezvous)")
    parser.add_argument("--n_eval_rollouts", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--results_dir", default=None, help="Path to population results root for policy-code teammate evaluation")
    args = parser.parse_args()

    env_module = importlib.import_module(f"popgen.envs.{args.env}.utils")

    if args.results_dir:
        _log_section("SCRIPTED POLICIES EVALUATION")
        summary = CooperatorEvaluator(
            cooperator_dir=args.cooperator_dir,
            env_module=env_module,
            eval_type="eval_teammate",
            num_workers=args.num_workers,
        ).evaluate(n_eval_rollouts=args.n_eval_rollouts)
        _log_scripted_policies_eval_summary(summary)

        _log_section("TRAINING TEAMMATES EVALUATION")
        et_summary = CooperatorEvaluator(
            cooperator_dir=args.cooperator_dir,
            env_module=env_module,
            eval_type="policy_code",
            population_dir=args.results_dir,
            num_workers=args.num_workers,
        ).evaluate(n_eval_rollouts=args.n_eval_rollouts, output_filename="training_teammates.json")
        logging.info(f"Teammates evaluated: {et_summary['n_teammates']}")
        logging.info(f"Rollouts per teammate: {et_summary['n_eval_rollouts']}")
        agg = et_summary["aggregate_reward_summary"]
        logging.info(f"Aggregate reward mean: {agg['mean']:.4f}  std: {agg['std']:.4f}")
        logging.info(f"Summary saved to: {et_summary['summary_path']}")
    else:
        _log_section("EVAL TEAMMATES EVALUATION")
        summary = CooperatorEvaluator(
            cooperator_dir=args.cooperator_dir,
            env_module=env_module,
            eval_type="eval_teammate",
            num_workers=args.num_workers,
        ).evaluate(n_eval_rollouts=args.n_eval_rollouts)
        _log_scripted_policies_eval_summary(summary)


# Run from the repository root, for example:
#   uv run python scripts/cooperator/gather_results.py \
#       --cooperator_dir /path/to/cooperator/output \
#       --env lbf \
#       --n_eval_rollouts 50 \
#       --results_dir /path/to/training/population
#
# Omit --results_dir to evaluate only against the environment's scripted eval
# teammates.
if __name__ == "__main__":
    main()
