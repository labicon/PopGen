import importlib
import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from popgen.cooperator import CooperatorTrainingPipeline, select_policy_refs_from_behavior_results

_SELF_PLAY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "self-play"
if str(_SELF_PLAY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_PLAY_SCRIPT_DIR))

from cooperator_training import resolve_cooperator_restart  # noqa: E402

if not OmegaConf.has_resolver("basename"):
    OmegaConf.register_new_resolver("basename", lambda s: s.split("/")[-1])


# Set log levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# load env variables from .env file
from dotenv import load_dotenv

load_dotenv(override=True)


def _resolve_input_path(path_str: str) -> Path:
    """Resolve config path robustly across invocation directories.

    Resolution order for relative paths:
    1. Hydra's original cwd (`to_absolute_path`)
    2. Project root inferred from this file location
    """
    path = Path(path_str)
    if path.is_absolute():
        return path

    hydra_resolved = Path(to_absolute_path(path_str))
    if hydra_resolved.exists():
        return hydra_resolved

    project_root = Path(__file__).resolve().parents[2]
    root_resolved = project_root / path
    if root_resolved.exists():
        return root_resolved

    return hydra_resolved


def run_cooperator_pipeline(cfg: DictConfig):
    """Run cooperator training from pre-generated behavior policy/reward results."""
    run_name = cfg.cooperator.run_name
    extended_output_dir = os.path.join(cfg.cooperator.output_dir, run_name)
    results_dir = _resolve_input_path(str(cfg.cooperator.results_dir))

    train_kwargs = OmegaConf.to_container(cfg.cooperator.train_kwargs, resolve=True)
    restart_from = cfg.cooperator.get("restart_from")
    restore_file, restart_output_dir = resolve_cooperator_restart(restart_from)
    if restore_file is not None:
        existing_restore = train_kwargs.get("restore_file")
        if existing_restore not in (None, restore_file):
            raise ValueError(
                f"Conflicting restore targets: cooperator.restart_from resolved to {restore_file}, "
                f"but cooperator.train_kwargs.restore_file is {existing_restore}."
            )
        train_kwargs["restore_file"] = restore_file
        extended_output_dir = restart_output_dir
        logging.info("Restarting from checkpoint=%s into output_dir=%s", restore_file, extended_output_dir)

    os.makedirs(extended_output_dir, exist_ok=True)
    logging.info(f"COOPERATOR_OUTPUT_DIR: {extended_output_dir}")

    logging.info("=" * 60)
    logging.info("COOPERATOR TRAINING PIPELINE")
    logging.info("=" * 60)
    logging.info(f"Environment: {cfg.population.env}")
    logging.info(f"Results:     {results_dir}")
    logging.info(f"Output:      {extended_output_dir}")
    logging.info("-" * 60)

    # Load env utils module (must expose create_env and train_fn)
    env_module = importlib.import_module(f"popgen.envs.{cfg.population.env}.utils")
    if not hasattr(env_module, "create_env"):
        raise AttributeError(f"popgen.envs.{cfg.population.env}.utils must define create_env")
    if not hasattr(env_module, "train_fn"):
        raise AttributeError(f"popgen.envs.{cfg.population.env}.utils must define train_fn")

    selected_refs = select_policy_refs_from_behavior_results(results_dir)
    logging.info(f"Selected teammate artifacts: {len(selected_refs)}")

    pipeline = CooperatorTrainingPipeline(
        env_module=env_module,
        results_dir=results_dir,
        output_dir=extended_output_dir,
        run_name=run_name,
        train_kwargs=train_kwargs,
        sum_team_reward=getattr(cfg.cooperator, "sum_team_reward", True),
    )
    experiment = pipeline.train_cooperator()

    logging.info("-" * 60)
    logging.info("COOPERATOR TRAINING COMPLETE")
    logging.info(f"Experiment: {type(experiment).__name__}")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_cooperator_pipeline(cfg)


# Run from the repository root with Hydra overrides, for example:
#   uv run python scripts/cooperator/generate_cooperator.py \
#       population.env=lbf \
#       population.output_dir=/path/to/population \
#       cooperator.results_dir=/path/to/population \
#       cooperator.run_name=lbf-cooperator
#
# The wrapper script `cd scripts/cooperator && ./run_cooperator.sh env=lbf
# population_dir=/path/to/population` runs this training step and then evaluates
# the resulting checkpoint.
#
# To resume training, pass cooperator.restart_from with either a checkpoint file
# or a run directory containing checkpoints. A checkpoint-file resume looks like:
#   uv run python scripts/cooperator/generate_cooperator.py \
#       population.env=overcooked \
#       population.output_dir=scripts/cooperator/results/overcooked/policy_code/gemini-3-flash-preview/verbalized_sampling/n4/20260429_012523 \
#       cooperator.results_dir=scripts/cooperator/results/overcooked/policy_code/gemini-3-flash-preview/verbalized_sampling/n4/20260429_012523 \
#       cooperator.restart_from=scripts/cooperator/results/overcooked/policy_code/gemini-3-flash-preview/verbalized_sampling/n4/20260429_012523/cooperator/overcooked-cooperator-20260429_012523/ippo_overcooked_gru__9c74866d_26_04_29-01_31_12/checkpoints/checkpoint_1200000.pt
#
# When cooperator.restart_from is set, the script restores that checkpoint and
# continues writing into the resolved cooperator run directory.
if __name__ == "__main__":
    main()
