import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from pipeline_utils import resolve_seeds, run_self_play_population

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def run_self_play_pipeline(cfg: DictConfig) -> str:
    env_name: str = cfg.env
    seeds = resolve_seeds(cfg.self_play.seeds, cfg.self_play.n_seeds)
    run_dir = to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))
    train_kwargs = OmegaConf.to_container(cfg.self_play.train_kwargs, resolve=True)

    logging.info(f"Self-play pipeline | env={env_name} | seeds={seeds} | output={run_dir}")
    run_self_play_population(
        env_name=env_name,
        output_dir=run_dir,
        seeds=seeds,
        n_workers=cfg.self_play.n_workers,
        train_kwargs=train_kwargs,
    )
    return run_dir


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_self_play_pipeline(cfg)


if __name__ == "__main__":
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    main()
