"""Run baseline evaluation (null-action ego) against scripted policies.

Results are saved under scripts/cooperator/results/baseline/<env_name>/ and can
be compared directly against CooperatorEvaluator output to measure how much the
cooperator contributes beyond what the teammate achieves on its own.

Usage:
    python scripts/cooperator/run_baseline_eval.py --env lbf
    python scripts/cooperator/run_baseline_eval.py --env point_mass_rendezvous --n_eval_rollouts 100
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from popgen.cooperator import BaselineEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate null-action ego against scripted policies to establish a teammate-only baseline."
    )
    parser.add_argument("--env", required=True, help="Environment name (e.g. lbf, point_mass_rendezvous)")
    parser.add_argument("--n_eval_rollouts", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = _SCRIPT_DIR / "results" / "baseline" / args.env
    output_dir.mkdir(parents=True, exist_ok=True)

    env_module = importlib.import_module(f"popgen.envs.{args.env}.utils")
    evaluator = BaselineEvaluator(
        output_dir=output_dir,
        env_module=env_module,
        num_workers=args.num_workers,
    )

    logging.info("=" * 60)
    logging.info("BASELINE SCRIPTED POLICIES EVALUATION")
    logging.info("=" * 60)
    logging.info(f"Env:              {args.env}")
    logging.info(f"Output dir:       {output_dir}")
    logging.info(f"Rollouts/policy:  {args.n_eval_rollouts}")

    summary = evaluator.evaluate_eval_teammates(n_eval_rollouts=args.n_eval_rollouts)

    logging.info("-" * 60)
    logging.info(f"Teammates:                 {summary['n_teammates']}")
    logging.info(f"Rollouts per teammate:     {summary['n_eval_rollouts']}")
    agg = summary["aggregate_reward_summary"]
    logging.info(f"Aggregate reward mean/std: {agg['mean']:.4f} / {agg['std']:.4f}")
    logging.info(f"Summary saved to:          {summary['summary_path']}")


# Run from the repository root, for example:
#   uv run python scripts/cooperator/run_baseline_eval.py --env lbf
#   uv run python scripts/cooperator/run_baseline_eval.py \
#       --env point_mass_rendezvous --n_eval_rollouts 100
#
# This evaluates a null-action ego policy against scripted teammates and writes
# results under scripts/cooperator/results/baseline/<env>/.
if __name__ == "__main__":
    main()
