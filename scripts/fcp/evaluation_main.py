import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

SELF_PLAY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "self-play"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (SELF_PLAY_SCRIPT_DIR, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from pipeline_utils import _enumerate_seed_dirs, resolve_env_module

from popgen.cooperator.evaluation import CooperatorEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def _resolve_existing_input_path(raw_path: str) -> Path:
    raw = str(raw_path).strip()
    if not raw:
        raise ValueError("Expected a non-empty path.")

    resolved = Path(to_absolute_path(raw)).expanduser().resolve()
    if resolved.exists():
        return resolved

    # Hydra treats values like "home/..." as relative paths. Recover the common
    # missing-leading-slash case before failing.
    if not raw.startswith(("/", ".", "..", "~")):
        rooted_candidate = Path(f"/{raw}").expanduser().resolve()
        if rooted_candidate.exists():
            logging.warning(
                "Interpreting path %r as %s because the provided value appears to be missing "
                "the leading '/'.",
                raw,
                rooted_candidate,
            )
            return rooted_candidate

    project_marker = f"/{PROJECT_ROOT.name}/"
    if project_marker in raw:
        suffix = raw.split(project_marker, 1)[1]
        remapped_candidate = (PROJECT_ROOT / suffix).resolve()
        if remapped_candidate.exists():
            logging.warning(
                "Interpreting path %r as %s because the original checkout root is not present "
                "on this machine.",
                raw,
                remapped_candidate,
            )
            return remapped_candidate

    return resolved


def run_fcp_evaluation(cfg: DictConfig) -> Path:
    if not cfg.fcp.self_play_results_dir:
        raise ValueError("fcp.self_play_results_dir must be set.")
    if not cfg.evaluation.cooperator_dir:
        raise ValueError("evaluation.cooperator_dir must be set.")

    env_module = resolve_env_module(cfg.env, required_attrs=("rollout_fn", "create_env"))
    self_play_results_dir = _resolve_existing_input_path(str(cfg.fcp.self_play_results_dir))
    cooperator_dir = _resolve_existing_input_path(str(cfg.evaluation.cooperator_dir))
    num_workers = cfg.evaluation.get("num_workers", 8)
    n_eval_rollouts = cfg.evaluation.n_eval_rollouts

    logging.info("=" * 60)
    logging.info("STAGE 1: SELF-PLAY TRAINING TEAMMATES EVALUATION")
    logging.info("=" * 60)
    teammate_summary = CooperatorEvaluator(
        cooperator_dir=cooperator_dir,
        env_module=env_module,
        eval_type="self_play",
        population_dir=self_play_results_dir,
        num_workers=num_workers,
    ).evaluate(n_eval_rollouts=n_eval_rollouts, output_filename="training_teammates.json")
    logging.info("Teammates:     %s", teammate_summary["n_teammates"])
    logging.info("Summary saved: %s", teammate_summary["summary_path"])

    logging.info("=" * 60)
    logging.info("STAGE 2: EVAL TEAMMATES EVALUATION")
    logging.info("=" * 60)
    scripted_summary = CooperatorEvaluator(
        cooperator_dir=cooperator_dir,
        env_module=env_module,
        eval_type="eval_teammate",
        num_workers=num_workers,
    ).evaluate(n_eval_rollouts=n_eval_rollouts)
    logging.info("Policies:      %s", scripted_summary["n_teammates"])
    logging.info("Summary saved: %s", scripted_summary["summary_path"])

    eval_dir = cooperator_dir / cfg.evaluation.output_subdir
    logging.info("=" * 60)
    logging.info("DONE — results at %s", eval_dir)
    logging.info("=" * 60)
    return eval_dir


@hydra.main(version_base=None, config_path="cfg", config_name="eval_config")
def main(cfg: DictConfig):
    """Evaluate an FCP cooperator against its checkpoint-pool training teammates and eval scripted teammates.

    Run from this directory:
        python evaluation_main.py env=lbf fcp.self_play_results_dir=/path/to/sp_run evaluation.cooperator_dir=/path/to/fcp_cooperator

    Key config fields (cfg/eval_config.yaml):
        env                              Environment name (lbf, football, smacv2, ...)
        fcp.self_play_results_dir        Path to the self-play run used during FCP training. Required.
        evaluation.cooperator_dir        Path to the FCP cooperator checkpoint directory. Required.
        evaluation.n_eval_rollouts       Number of rollouts per teammate (default: 50)
        evaluation.num_workers           Parallel rollout workers (default: 8)
        evaluation.seed                  Base random seed for rollouts (default: null)
    """
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_fcp_evaluation(cfg)


if __name__ == "__main__":
    main()
