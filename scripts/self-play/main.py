import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from cooperator_evaluation import run_cooperator_evaluation_pipeline
from cooperator_training import train_cooperator_against_population
from pipeline_utils import resolve_seeds
from population_generation import run_self_play_population

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def run_full_pipeline(cfg: DictConfig):
    env_name = cfg.env
    cooperator_train_kwargs = OmegaConf.to_container(cfg.cooperator.train_kwargs, resolve=True)
    run_self_play = cfg.pipeline.get("run_self_play", True)

    if run_self_play:
        seeds = resolve_seeds(cfg.self_play.seeds, cfg.self_play.n_seeds, cfg.self_play.population_seed)
        run_dir = to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))
        self_play_train_kwargs = OmegaConf.to_container(cfg.self_play.train_kwargs, resolve=True)

        logging.info("=" * 60)
        logging.info("STAGE 1: SELF-PLAY POPULATION GENERATION")
        logging.info("=" * 60)
        logging.info(
            "Self-play config | population_seed=%s | resolved_seeds=%s",
            cfg.self_play.population_seed,
            seeds,
        )
        run_self_play_population(
            env_name=env_name,
            output_dir=run_dir,
            seeds=seeds,
            n_workers=cfg.self_play.n_workers,
            train_kwargs=self_play_train_kwargs,
        )
    else:
        if not cfg.cooperator.self_play_results_dir:
            raise ValueError("pipeline.run_self_play=false requires cooperator.self_play_results_dir to be set.")
        run_dir = to_absolute_path(cfg.cooperator.self_play_results_dir)
        logging.info("=" * 60)
        logging.info("SKIPPING STAGE 1: using existing self-play population at %s", run_dir)
        logging.info("=" * 60)

    cooperator_output_dir = os.path.join(run_dir, cfg.cooperator.output_subdir, cfg.cooperator.run_name)

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
        run_cooperator_evaluation_pipeline(
            run_dir=run_dir,
            env_name=env_name,
            n_eval_rollouts=cfg.evaluation.n_eval_rollouts,
            num_workers=cfg.evaluation.num_workers,
            seed=cfg.evaluation.seed,
        )

    logging.info("=" * 60)
    logging.info("DONE")
    logging.info(f"Self-play seeds: {run_dir}/seed_*/")
    logging.info(f"Cooperator:      {run_dir}/{cfg.cooperator.output_subdir}/")
    logging.info("=" * 60)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    """Run the full self-play pipeline: population generation → cooperator training → evaluation.

    Run from this directory:
        python main.py env=lbf
        python main.py env=lbf pipeline.run_self_play=false cooperator.self_play_results_dir=/path/to/sp_run

    Key config fields (cfg/config.yaml):
        env                               Environment name (lbf, football, smacv2, ...)
        pipeline.run_self_play            Run self-play stage; set false to reuse an existing population (default: true)
        pipeline.run_evaluation           Run evaluation after cooperator training (default: true)
        self_play.n_seeds                 Number of independent self-play seeds (default: 4)
        self_play.output_dir              Root directory for self-play results
        self_play.run_name                Subdirectory name for this run (default: timestamp)
        self_play.train_kwargs            Extra kwargs forwarded to BenchMARL self-play training
        cooperator.self_play_results_dir  Existing population dir (required when pipeline.run_self_play=false)
        cooperator.train_kwargs           Extra kwargs forwarded to cooperator training
        evaluation.n_eval_rollouts        Rollouts per teammate for the evaluation stage (default: 50)
        evaluation.num_workers            Parallel rollout workers (default: 8)
    """
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
