import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from evaluation import run_evaluation_pipeline
from pipeline_utils import resolve_seeds, run_self_play_population, train_cooperator_against_population

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def run_full_pipeline(cfg: DictConfig):
    env_name = cfg.env
    seeds = resolve_seeds(cfg.self_play.seeds, cfg.self_play.n_seeds)
    run_dir = to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))
    self_play_train_kwargs = OmegaConf.to_container(cfg.self_play.train_kwargs, resolve=True)
    cooperator_train_kwargs = OmegaConf.to_container(cfg.cooperator.train_kwargs, resolve=True)
    cooperator_output_dir = os.path.join(run_dir, cfg.cooperator.output_subdir, cfg.cooperator.run_name)

    logging.info("=" * 60)
    logging.info("STAGE 1: SELF-PLAY POPULATION GENERATION")
    logging.info("=" * 60)
    run_self_play_population(
        env_name=env_name,
        output_dir=run_dir,
        seeds=seeds,
        n_workers=cfg.self_play.n_workers,
        train_kwargs=self_play_train_kwargs,
    )

    logging.info("=" * 60)
    logging.info("STAGE 2: COOPERATOR TRAINING")
    logging.info("=" * 60)
    train_cooperator_against_population(
        env_name=env_name,
        self_play_results_dir=run_dir,
        output_dir=cooperator_output_dir,
        train_kwargs=cooperator_train_kwargs,
    )

    if cfg.pipeline.run_evaluation:
        logging.info("=" * 60)
        logging.info("STAGE 3+: EVALUATION")
        logging.info("=" * 60)
        run_evaluation_pipeline(
            run_dir=run_dir,
            env_name=env_name,
            n_eval_rollouts=cfg.evaluation.n_eval_rollouts,
            seed=cfg.evaluation.seed,
        )

    logging.info("=" * 60)
    logging.info("DONE")
    logging.info(f"Self-play seeds: {run_dir}/seed_*/")
    logging.info(f"Cooperator:      {run_dir}/{cfg.cooperator.output_subdir}/")
    logging.info("=" * 60)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
