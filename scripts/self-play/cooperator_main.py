import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from pipeline_utils import train_cooperator_against_population

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def resolve_self_play_results_dir(cfg: DictConfig) -> str:
    if cfg.cooperator.self_play_results_dir is not None:
        return to_absolute_path(str(cfg.cooperator.self_play_results_dir))
    return to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))


def run_self_play_cooperator(cfg: DictConfig) -> str:
    results_dir = resolve_self_play_results_dir(cfg)
    output_dir = os.path.join(results_dir, cfg.cooperator.output_subdir, cfg.cooperator.run_name)
    train_kwargs = OmegaConf.to_container(cfg.cooperator.train_kwargs, resolve=True)

    logging.info(f"Cooperator pipeline | env={cfg.env} | results={results_dir} | output={output_dir}")
    train_cooperator_against_population(
        env_name=cfg.env,
        self_play_results_dir=results_dir,
        output_dir=output_dir,
        train_kwargs=train_kwargs,
    )
    return output_dir


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_self_play_cooperator(cfg)


if __name__ == "__main__":
    main()
