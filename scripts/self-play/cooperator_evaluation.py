import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_utils import resolve_env_module

from popgen.cooperator.evaluation import CooperatorEvaluator, _resolve_scripted_policies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def find_cooperator_dir(run_dir: str | Path) -> Path:
    run_path = Path(run_dir).resolve()
    cooperator_root = run_path / "cooperator"
    if not cooperator_root.exists():
        raise FileNotFoundError(f"Cooperator directory does not exist: {cooperator_root}")
    cooperator_runs = [p for p in cooperator_root.iterdir() if p.is_dir()]
    if not cooperator_runs:
        raise FileNotFoundError(f"No cooperator runs found under {cooperator_root}")
    return max(cooperator_runs, key=lambda p: p.name)


def run_cooperator_evaluation_pipeline(
    run_dir: str | Path,
    env_name: str,
    n_eval_rollouts: int,
    num_workers: int,
    seed: int | None = None,
) -> Path:
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_path}")

    env_module = resolve_env_module(env_name, required_attrs=("rollout_fn", "create_env"))
    cooperator_dir = find_cooperator_dir(run_path)

    logging.info("=" * 60)
    logging.info("SELF-PLAY TRAINING TEAMMATES EVALUATION")
    logging.info("=" * 60)
    teammate_summary = CooperatorEvaluator(
        cooperator_dir=cooperator_dir,
        env_module=env_module,
        eval_type="self_play",
        population_dir=run_path,
        num_workers=num_workers,
    ).evaluate(n_eval_rollouts=n_eval_rollouts, output_filename="training_teammates.json")
    logging.info("Teammates:     %s", teammate_summary["n_teammates"])
    logging.info("Summary saved: %s", teammate_summary["summary_path"])

    try:
        _resolve_scripted_policies(env_module, "standard")
    except AttributeError:
        logging.info("Skipping scripted evaluation (env module defines no scripted evaluation set).")
    else:
        logging.info("=" * 60)
        logging.info("EVAL TEAMMATES EVALUATION")
        logging.info("=" * 60)
        scripted_summary = CooperatorEvaluator(
            cooperator_dir=cooperator_dir,
            env_module=env_module,
            eval_type="eval_teammate",
            num_workers=num_workers,
        ).evaluate(n_eval_rollouts=n_eval_rollouts)
        logging.info("Policies:      %s", scripted_summary["n_teammates"])
        logging.info("Summary saved: %s", scripted_summary["summary_path"])

    eval_dir = cooperator_dir / "evaluation"
    logging.info("=" * 60)
    logging.info("DONE — results at %s", eval_dir)
    logging.info("=" * 60)
    return eval_dir


def resolve_cooperator_evaluation_run_dir(cfg: DictConfig) -> str:
    if cfg.evaluation.run_dir is not None:
        return to_absolute_path(str(cfg.evaluation.run_dir))
    if cfg.cooperator.self_play_results_dir is not None:
        return to_absolute_path(str(cfg.cooperator.self_play_results_dir))
    return to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    """Evaluate a self-play cooperator against its training teammates and eval scripted teammates.

    Run from this directory:
        python cooperator_evaluation.py env=lbf evaluation.run_dir=/path/to/sp_run
        python cooperator_evaluation.py env=lbf cooperator.self_play_results_dir=/path/to/sp_run

    Key config fields (cfg/config.yaml):
        env                               Environment name (lbf, football, smacv2, ...)
        evaluation.run_dir                Path to the self-play run directory (seed_*/ + cooperator/).
                                          If unset, falls back to cooperator.self_play_results_dir,
                                          then to <self_play.output_dir>/<self_play.run_name>.
        cooperator.self_play_results_dir  Alternative path to the self-play run directory.
        evaluation.n_eval_rollouts        Number of rollouts per teammate (default: 50)
        evaluation.num_workers            Parallel rollout workers (default: 8)
        evaluation.seed                   Base random seed for rollouts (default: null)
    """
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_cooperator_evaluation_pipeline(
        run_dir=resolve_cooperator_evaluation_run_dir(cfg),
        env_name=cfg.env,
        n_eval_rollouts=cfg.evaluation.n_eval_rollouts,
        num_workers=cfg.evaluation.num_workers,
        seed=cfg.evaluation.seed,
    )


if __name__ == "__main__":
    main()
