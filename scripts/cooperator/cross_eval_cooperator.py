"""Cross-evaluate an existing cooperator against a non-training teammate population.

Usage:
    uv run python scripts/cooperator/cross_eval_cooperator.py \
        --env lbf \
        --cooperator_dir /path/to/cooperator_dir \
        --population_dir /path/to/policy_population \
        --population_type policy_code

    uv run python scripts/cooperator/cross_eval_cooperator.py \
        --env lbf \
        --run_dir /path/to/self_play_run \
        --population_dir /path/to/policy_population \
        --population_type policy_code
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from popgen.utils.filesystem import find_reward_checkpoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def _contains_reward_checkpoints(path: Path) -> bool:
    return bool(find_reward_checkpoints(path))


def _is_cooperator_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "evaluation").is_dir() and _contains_reward_checkpoints(path)


def _select_latest_candidate(path: Path, *, skip_seed_dirs: bool = False) -> Path | None:
    candidates: list[Path] = []
    for child in sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name):
        if child.name.startswith("."):
            continue
        if skip_seed_dirs and child.name.startswith("seed_"):
            continue
        if _is_cooperator_run_dir(child) or _contains_reward_checkpoints(child):
            candidates.append(child)
    if not candidates:
        return None
    return candidates[-1]


def _find_latest_cooperator_dir(run_dir: str | Path) -> Path:
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_path}")

    cooperator_root = run_path / "cooperator"
    if cooperator_root.is_dir():
        if _is_cooperator_run_dir(cooperator_root):
            return cooperator_root

        latest_nested = _select_latest_candidate(cooperator_root)
        if latest_nested is not None:
            return latest_nested

        if _contains_reward_checkpoints(cooperator_root):
            return cooperator_root

        raise FileNotFoundError(
            f"No cooperator runs with checkpoints found under {cooperator_root}"
        )

    if _is_cooperator_run_dir(run_path):
        return run_path

    latest_direct = _select_latest_candidate(run_path, skip_seed_dirs=True)
    if latest_direct is not None:
        return latest_direct

    if _contains_reward_checkpoints(run_path):
        return run_path

    raise FileNotFoundError(
        "Could not resolve a cooperator directory from run_dir: "
        f"{run_path}. Expected either run_dir/cooperator/<run>/..., "
        "or a direct child containing checkpoints (for fcp-style runs)."
    )


def _resolve_cooperator_dir(args: argparse.Namespace) -> Path:
    if args.cooperator_dir is not None:
        return Path(args.cooperator_dir).resolve()
    return _find_latest_cooperator_dir(args.run_dir)


def _default_output_filename(population_type: str) -> str:
    return f"cross_eval_{population_type}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate a trained cooperator against a policy-code or self-play population."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--cooperator_dir",
        help="Path to the cooperator output directory that contains checkpoints/.",
    )
    target_group.add_argument(
        "--run_dir",
        help=(
            "Higher-level run directory. Supports self-play/our-method layouts with "
            "cooperator/<run_name>/ and fcp-style layouts where the checkpoint directory "
            "is a direct child of run_dir; the latest valid run is selected."
        ),
    )
    parser.add_argument("--env", required=True, help="Environment name (e.g. lbf, pp, point_mass_rendezvous)")
    parser.add_argument(
        "--population_dir",
        required=True,
        help="Teammate population root. For policy_code this must contain agents/ and metadata/. For self_play it must contain seed_*/.",
    )
    parser.add_argument(
        "--population_type",
        choices=("policy_code", "self_play"),
        default="policy_code",
        help="Which teammate population format to evaluate against (default: policy_code).",
    )
    parser.add_argument(
        "--output_filename",
        default=None,
        help="Filename to write under <cooperator_dir>/evaluation/ (default: cross_eval_<population_type>.json).",
    )
    parser.add_argument("--n_eval_rollouts", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    from popgen.cooperator import CooperatorEvaluator

    cooperator_dir = _resolve_cooperator_dir(args)
    env_module = importlib.import_module(f"popgen.envs.{args.env}.utils")
    output_filename = args.output_filename or _default_output_filename(args.population_type)

    logging.info("=" * 60)
    logging.info("CROSS-EVALUATION")
    logging.info("=" * 60)
    logging.info("Env:              %s", args.env)
    logging.info("Cooperator dir:   %s", cooperator_dir)
    logging.info("Population dir:   %s", Path(args.population_dir).resolve())
    logging.info("Population type:  %s", args.population_type)
    logging.info("Rollouts/pop:     %d", args.n_eval_rollouts)
    logging.info("Output filename:  %s", output_filename)

    summary = CooperatorEvaluator(
        cooperator_dir=cooperator_dir,
        env_module=env_module,
        eval_type=args.population_type,
        population_dir=args.population_dir,
        num_workers=args.num_workers,
    ).evaluate(
        n_eval_rollouts=args.n_eval_rollouts,
        output_filename=output_filename,
    )

    logging.info("-" * 60)
    logging.info("Teammates:                  %d", summary["n_teammates"])
    logging.info("Rollouts per teammate:      %d", summary["n_eval_rollouts"])
    agg = summary["aggregate_reward_summary"]
    logging.info("Aggregate reward mean/std:  %.4f / %.4f", agg["mean"], agg["std"])
    logging.info("Summary saved to:           %s", summary["summary_path"])


# Run from the repository root, for example:
#   uv run python scripts/cooperator/cross_eval_cooperator.py \
#       --env lbf \
#       --cooperator_dir /path/to/cooperator/output \
#       --population_dir /path/to/heldout/population \
#       --population_type policy_code
#
# Use --run_dir instead of --cooperator_dir when you want the script to resolve
# the latest checkpoint directory inside a higher-level experiment run.
if __name__ == "__main__":
    main()
