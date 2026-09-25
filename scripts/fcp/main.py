import json
import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

SELF_PLAY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "self-play"
FCP_SCRIPT_DIR = Path(__file__).resolve().parent
if str(SELF_PLAY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_PLAY_SCRIPT_DIR))

from cooperator_training import resolve_cooperator_restart, train_fcp_against_population

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def _resolve_existing_input_path(raw_path: str) -> str:
    raw = str(raw_path).strip()
    if not raw:
        raise ValueError("Expected a non-empty path.")

    resolved = Path(to_absolute_path(raw)).resolve()
    if resolved.exists():
        return str(resolved)

    # Hydra treats values like "home/..." as relative paths. Recover the common
    # missing-leading-slash case before failing.
    if not raw.startswith(("/", ".", "..", "~")):
        rooted_candidate = Path(f"/{raw}").resolve()
        if rooted_candidate.exists():
            logging.warning(
                "Interpreting fcp.self_play_results_dir=%r as %s because the provided path "
                "appears to be missing the leading '/'.",
                raw,
                rooted_candidate,
            )
            return str(rooted_candidate)

    return str(resolved)


def _load_self_play_results_dir_from_population_json(run_dir: str | Path) -> str | None:
    population_json = Path(run_dir) / "fcp_population.json"
    if not population_json.exists():
        return None
    try:
        with population_json.open() as handle:
            return json.load(handle).get("self_play_results_dir")
    except Exception:
        return None


def run_fcp_pipeline(cfg: DictConfig) -> dict:
    restart_from = cfg.fcp.get("restart_from")
    restore_file, restart_output_dir = resolve_cooperator_restart(restart_from)

    train_kwargs = OmegaConf.to_container(cfg.fcp.train_kwargs, resolve=True)

    # Resolve self_play_results_dir — required unless restart supplies it via fcp_population.json.
    if cfg.fcp.self_play_results_dir:
        self_play_results_dir = _resolve_existing_input_path(str(cfg.fcp.self_play_results_dir))
    elif restart_output_dir is not None:
        inferred = _load_self_play_results_dir_from_population_json(restart_output_dir)
        if inferred is None:
            raise ValueError(
                "fcp.self_play_results_dir must be set when restarting without fcp_population.json."
            )
        self_play_results_dir = inferred
        logging.info("Inferred self_play_results_dir from fcp_population.json: %s", self_play_results_dir)
    else:
        raise ValueError("fcp.self_play_results_dir must be set to an existing self-play run directory.")

    # Resolve output_dir and inject restore checkpoint when restarting.
    if restart_output_dir is not None:
        output_dir = restart_output_dir
        existing_restore = train_kwargs.get("restore_file")
        if existing_restore not in (None, restore_file):
            raise ValueError(
                "Conflicting FCP restore targets: "
                f"fcp.restart_from resolved to {restore_file}, "
                f"but fcp.train_kwargs.restore_file is {existing_restore}."
            )
        train_kwargs["restore_file"] = restore_file
        logging.info(
            "Restarting FCP training from checkpoint=%s into output_dir=%s",
            restore_file,
            output_dir,
        )
    else:
        configured_output_root = Path(str(cfg.fcp.output_dir))
        if configured_output_root.is_absolute():
            output_root = configured_output_root
        else:
            output_root = FCP_SCRIPT_DIR / configured_output_root
        output_dir = str((output_root / str(cfg.fcp.run_name)).resolve())

    logging.info("=" * 60)
    logging.info("FICTITIOUS CO-PLAY STAGE 2")
    logging.info("=" * 60)
    logging.info("FCP source self-play experiment: %s", self_play_results_dir)
    logging.info(
        "FCP config | env=%s | self_play_results=%s | output=%s",
        cfg.env,
        self_play_results_dir,
        output_dir,
    )

    metadata = train_fcp_against_population(
        env_name=cfg.env,
        self_play_results_dir=self_play_results_dir,
        output_dir=output_dir,
        train_kwargs=train_kwargs,
        max_checkpoints_per_seed=cfg.fcp.max_checkpoints_per_seed,
    )

    logging.info("=" * 60)
    logging.info("DONE")
    logging.info("FCP output:    %s", output_dir)
    logging.info("Pool metadata: %s", metadata["metadata_path"])
    logging.info(
        "Checkpoint pool: %d checkpoints across %d seeds",
        metadata["n_checkpoint_policies"],
        metadata["n_seed_runs"],
    )
    logging.info("=" * 60)
    return metadata


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    """Train an FCP cooperator against a self-play checkpoint pool.

    Run from this directory:
        python main.py env=lbf fcp.self_play_results_dir=/path/to/sp_run
        python main.py env=lbf fcp.self_play_results_dir=/path/to/sp_run fcp.max_checkpoints_per_seed=5
        python main.py env=lbf fcp.restart_from=/path/to/fcp_run
        python main.py env=lbf fcp.restart_from=/path/to/checkpoints/checkpoint_804000.pt

    Key config fields (cfg/config.yaml):
        env                              Environment name (lbf, football, smacv2, ...)
        fcp.self_play_results_dir        Path to an existing self-play run (seed_*/ dirs). Required
                                         unless fcp.restart_from is set and fcp_population.json exists.
        fcp.restart_from                 Optional checkpoint file, experiment dir, or FCP run dir
                                         to resume training from the latest checkpoint.
        fcp.max_checkpoints_per_seed     Max checkpoints to sample per seed; if exceeded, evenly spaced
                                         subset is used (default: 10)
        fcp.output_dir                   Root directory for FCP results (default: results/<env>/fcp)
        fcp.run_name                     Subdirectory name for this run (default: timestamp)
        fcp.train_kwargs                 Extra kwargs forwarded to env_module.train_fn
    """
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_fcp_pipeline(cfg)


if __name__ == "__main__":
    main()
