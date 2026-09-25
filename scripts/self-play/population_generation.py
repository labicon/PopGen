import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
from benchmarl.experiment import Experiment, ExperimentConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pipeline_utils import (
    _build_task,
    _checkpoint_step,
    _noop_context,
    _pp_semdiv_context,
    _pp_stag_semdiv_context,
    resolve_env_module,
)
from pqdm.processes import pqdm

from popgen.envs.utils import load_algorithm_config, load_model_config, resolve_cfg_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def _find_latest_selfplay_checkpoint(seed_output_dir: str) -> Path | None:
    checkpoints = sorted(
        Path(seed_output_dir).rglob("checkpoints/checkpoint_*.pt"),
        key=_checkpoint_step,
    )
    return checkpoints[-1] if checkpoints else None


def _train_single_seed(args: tuple[int, str, str, dict[str, Any], str | None]):
    seed, seed_output_dir, env_name, train_kwargs, restore_file = args
    env_module = resolve_env_module(env_name, required_attrs=("RL_CFG_DIR",))
    base_rl_cfg_dir: Path = env_module.RL_CFG_DIR

    os.makedirs(seed_output_dir, exist_ok=True)

    if env_name == "pp":
        train_ctx = _pp_semdiv_context()
    elif env_name == "pp_stag":
        train_ctx = _pp_stag_semdiv_context()
    else:
        train_ctx = _noop_context()

    wandb_kwargs = {"name": f"{env_name}_selfplay_seed_{seed}", "tags": [env_name, "self_play"]}

    rl_cfg_dir = resolve_cfg_dir(base_rl_cfg_dir, teammate_policy=None)
    task = _build_task(env_name, base_rl_cfg_dir)
    algorithm_config = load_algorithm_config(rl_cfg_dir)
    experiment_config = ExperimentConfig.get_from_yaml(path=str(rl_cfg_dir / "base_experiment.yaml"))
    model_cfg_name = train_kwargs.get("model_config_name")
    model_config = load_model_config(rl_cfg_dir, name=model_cfg_name)
    try:
        critic_model_config = load_model_config(rl_cfg_dir, prefix="critic_", name=model_cfg_name)
    except FileNotFoundError:
        critic_model_config = load_model_config(rl_cfg_dir, name=model_cfg_name)

    for key, value in train_kwargs.items():
        setattr(experiment_config, key, value)

    experiment_config.save_folder = seed_output_dir
    experiment_config.wandb_extra_kwargs = wandb_kwargs

    if restore_file is not None:
        logging.info("[seed=%s] Resuming from checkpoint: %s", seed, restore_file)
        experiment_config.restore_file = restore_file

    with train_ctx:
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=experiment_config,
        )
        experiment.run()

    logging.info("[seed=%s] Training complete -> %s", seed, seed_output_dir)
    return seed_output_dir


def _infer_seeds_from_dir(run_dir: str) -> list[int]:
    seed_dirs = sorted(Path(run_dir).glob("seed_*/"), key=lambda p: p.name)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_*/ subdirectories found in: {run_dir}")
    seeds = []
    for seed_dir in seed_dirs:
        try:
            seeds.append(int(seed_dir.name.removeprefix("seed_")))
        except ValueError:
            logging.warning("Ignoring unexpected directory: %s", seed_dir.name)
    return seeds


def run_self_play_population(
    env_name: str,
    output_dir: str,
    seeds: Sequence[int],
    n_workers: int,
    train_kwargs: dict[str, Any],
    restart_from: str | None = None,
) -> list[str]:
    resolve_env_module(env_name, required_attrs=("RL_CFG_DIR",))
    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for seed in seeds:
        seed_output_dir = os.path.join(output_dir, f"seed_{int(seed)}")
        restore_file = None
        if restart_from is not None:
            checkpoint = _find_latest_selfplay_checkpoint(os.path.join(restart_from, f"seed_{int(seed)}"))
            if checkpoint is not None:
                restore_file = str(checkpoint)
                logging.info("[seed=%s] Will resume from checkpoint: %s", seed, restore_file)
            else:
                logging.info("[seed=%s] No checkpoint found; starting from scratch.", seed)
        jobs.append((int(seed), seed_output_dir, env_name, dict(train_kwargs), restore_file))

    results = pqdm(jobs, _train_single_seed, n_jobs=n_workers, desc="Self-play training")
    failures = [result for result in results if isinstance(result, Exception)]
    if failures:
        raise RuntimeError(f"{len(failures)} self-play seeds failed. First error: {failures[0]}")
    return [str(result) for result in results]


def run_self_play_pipeline(cfg: DictConfig) -> str:
    from pipeline_utils import resolve_seeds

    env_name: str = cfg.env
    train_kwargs = OmegaConf.to_container(cfg.self_play.train_kwargs, resolve=True)
    restart_from = cfg.self_play.get("restart_from")

    if restart_from is not None:
        restart_from = to_absolute_path(str(restart_from))
        if not Path(restart_from).exists():
            raise FileNotFoundError(f"Self-play restart path does not exist: {restart_from}")
        run_dir = restart_from
        seeds = _infer_seeds_from_dir(restart_from)
        logging.info(
            "Self-play pipeline | RESUMING | env=%s | restart_from=%s | seeds=%s",
            env_name,
            restart_from,
            seeds,
        )
    else:
        seeds = resolve_seeds(cfg.self_play.seeds, cfg.self_play.n_seeds, cfg.self_play.population_seed)
        run_dir = to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))
        logging.info(
            "Self-play pipeline | env=%s | population_seed=%s | seeds=%s | output=%s",
            env_name,
            cfg.self_play.population_seed,
            seeds,
            run_dir,
        )

    run_self_play_population(
        env_name=env_name,
        output_dir=run_dir,
        seeds=seeds,
        n_workers=cfg.self_play.n_workers,
        train_kwargs=train_kwargs,
        restart_from=restart_from,
    )
    return run_dir


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    """Train a self-play population across multiple seeds.

    Run from this directory:
        python population_generation.py env=lbf
        python population_generation.py env=lbf self_play.n_seeds=8
        python population_generation.py env=pp_stag self_play.restart_from=/path/to/existing/run

    Key config fields (cfg/config.yaml):
        env                           Environment name (lbf, football, smacv2, ...)
        self_play.n_seeds             Number of independent self-play training seeds (default: 4)
        self_play.seeds               Explicit seed list; overrides n_seeds when set
        self_play.population_seed     RNG seed used to sample seeds when self_play.seeds is null
        self_play.n_workers           Parallel workers (set >1 only with multiple GPUs)
        self_play.output_dir          Root directory for results (default: ./results/<env>/self_play)
        self_play.run_name            Subdirectory name for this run (default: timestamp)
        self_play.train_kwargs        Extra kwargs forwarded to BenchMARL training
        self_play.restart_from        Path to an existing run directory to resume. Seeds are
                                      inferred from seed_*/ subdirectories; each seed resumes
                                      from its latest checkpoint (or trains from scratch if none).
                                      run_name and output_dir are ignored when this is set.
    """
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_self_play_pipeline(cfg)


if __name__ == "__main__":
    main()
