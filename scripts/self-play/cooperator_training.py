import json
import logging
import os
import pickle
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
from benchmarl.experiment import Experiment
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pipeline_utils import _checkpoint_step, _enumerate_seed_dirs, resolve_env_module

from popgen.cooperator.selfplay_util import (
    SelfPlayPolicyTensorAdapter,
    SelfPlayPolicyWrapper,
    _make_stub_unpickler,
    close_env_safely,
    load_selfplay_policy,
)
from popgen.utils.checkpoints import select_evenly_spaced_paths
from popgen.utils.run_names import build_cooperator_run_name, extract_run_date_from_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_latest_policy_and_env(
    save_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    allow_dynamic_stubs: bool = False,
    move_policy_to_cpu: bool = False,
):
    """Load the latest (or a specific) BenchMARL checkpoint from a seed directory.

    Returns ``(policy, env, policy_group)``.
    """
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}")
    else:
        if save_dir is None:
            raise ValueError("Either save_dir or checkpoint_path must be provided.")
        checkpoints = sorted(
            Path(save_dir).rglob("checkpoints/checkpoint_*.pt"),
            key=_checkpoint_step,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {save_dir}")
        checkpoint = checkpoints[-1]

    experiment_folder = checkpoint.parent.parent.resolve()
    original_pickle_load = pickle.load

    if allow_dynamic_stubs:
        stub_unpickler = _make_stub_unpickler()

        def _patched_load(file_obj, **kwargs):
            return stub_unpickler(file_obj).load()

        pickle.load = _patched_load

    try:
        experiment = Experiment.reload_from_file(
            str(checkpoint),
            experiment_patch={
                "save_folder": str(experiment_folder.parent),
                "loggers": [],
                "create_json": False,
                "sampling_device": "cuda",
                "train_device": "cuda",
                "buffer_device": "cuda",
                "restore_map_location": "cuda",
                "parallel_collection": False,
                "collect_with_grad": True,
                "evaluation": False,
                "render": False,
                "evaluation_episodes": 1,
                "on_policy_n_envs_per_worker": 1,
                "off_policy_n_envs_per_worker": 1,
                "on_policy_collected_frames_per_batch": 1,
                "off_policy_collected_frames_per_batch": 1,
            },
        )
    finally:
        pickle.load = original_pickle_load

    for candidate in ("ego", "agents"):
        if candidate in experiment.group_policies:
            policy_group = candidate
            break
    else:
        policy_group = next(iter(experiment.group_policies))
        logging.warning(
            "load_latest_policy_and_env: no 'ego'/'agents' group in %s; using %r",
            checkpoint,
            policy_group,
        )

    policy = experiment.group_policies[policy_group]
    if move_policy_to_cpu:
        policy = policy.cpu()
    return policy, experiment.test_env, policy_group


def _load_checkpoint_policy(
    checkpoint_path: Path,
    reload_context: Callable | None = None,
) -> SelfPlayPolicyWrapper:
    """Load a specific checkpoint (for FCP, which samples across training steps)."""
    seed_dir = next((p for p in checkpoint_path.parents if p.name.startswith("seed_")), None)
    if seed_dir is None:
        raise ValueError(f"Could not infer seed directory from checkpoint path {checkpoint_path}")

    logging.info("Loading self-play checkpoint policy from %s", checkpoint_path)
    ctx = reload_context() if callable(reload_context) else nullcontext()
    with ctx:
        raw_policy, env, trained_group = load_latest_policy_and_env(
            checkpoint_path=checkpoint_path,
            allow_dynamic_stubs=True,
            move_policy_to_cpu=False,
        )
    try:
        n_agents = env.observation_spec[trained_group, "observation"].shape[-2]
    except (KeyError, IndexError, AttributeError):
        n_agents = _get_n_agents_from_env(env, trained_group)
    close_env_safely(env)
    logging.info(
        "  trained_group=%r  n_agents_trained=%s  checkpoint_step=%s",
        trained_group,
        n_agents,
        _checkpoint_step(checkpoint_path),
    )
    return SelfPlayPolicyWrapper(raw_policy, trained_group=trained_group, n_agents_trained=n_agents, seed_dir=seed_dir)


def _get_n_agents_from_env(env, group: str) -> int:
    for attr in ("n_agents", "num_agents"):
        if hasattr(env, attr):
            return int(getattr(env, attr))
    try:
        return env.observation_spec[group, "observation"].shape[-2]
    except (KeyError, IndexError, AttributeError):
        pass
    raise RuntimeError(f"Could not determine n_agents from env {type(env).__name__}")


# ---------------------------------------------------------------------------
# Population discovery
# ---------------------------------------------------------------------------


def discover_self_play_checkpoints(
    self_play_results_dir: str | Path,
    max_checkpoints_per_seed: int | None = None,
) -> list[Path]:
    """Return a flat list of checkpoint paths across all seed directories.

    Used by FCP to sample policies at different training snapshots.
    """
    results_dir = Path(self_play_results_dir)
    seed_dirs = _enumerate_seed_dirs(results_dir)

    checkpoint_paths: list[Path] = []
    for seed_dir in seed_dirs:
        seed_checkpoints = sorted(
            seed_dir.rglob("checkpoints/checkpoint_*.pt"),
            key=lambda path: (path.parent.parent.relative_to(seed_dir).as_posix(), _checkpoint_step(path)),
        )
        if not seed_checkpoints:
            raise FileNotFoundError(f"No checkpoints found under self-play seed directory {seed_dir}")
        selected = select_evenly_spaced_paths(seed_checkpoints, max_checkpoints_per_seed)
        if len(selected) != len(seed_checkpoints):
            logging.info(
                "Subsampled checkpoints for %s: keeping %d of %d steps (%s)",
                seed_dir.name,
                len(selected),
                len(seed_checkpoints),
                ", ".join(str(_checkpoint_step(p)) for p in selected),
            )
        checkpoint_paths.extend(selected)

    if not checkpoint_paths:
        raise FileNotFoundError(f"No self-play checkpoints found in {results_dir}")
    return checkpoint_paths


def load_checkpoint_policies(
    checkpoint_paths: Sequence[str | Path],
    reload_context: Callable | None = None,
) -> list[SelfPlayPolicyWrapper]:
    """Load one SelfPlayPolicyWrapper per checkpoint path (for FCP)."""
    return [
        _load_checkpoint_policy(Path(p), reload_context=reload_context)
        for p in checkpoint_paths
    ]


# ---------------------------------------------------------------------------
# Cooperator training
# ---------------------------------------------------------------------------


def _find_latest_checkpoint_in_dir(path: Path) -> Path:
    checkpoints = list(path.rglob("checkpoints/checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {path}")

    experiment_dirs = {checkpoint.parent.parent.resolve() for checkpoint in checkpoints}
    if len(experiment_dirs) == 1:
        return max(checkpoints, key=lambda checkpoint: (_checkpoint_step(checkpoint), str(checkpoint.resolve())))

    logging.warning(
        "Found checkpoints under multiple experiment directories in %s; "
        "selecting the most recently modified checkpoint for restart.",
        path,
    )
    return max(
        checkpoints,
        key=lambda checkpoint: (
            checkpoint.stat().st_mtime,
            _checkpoint_step(checkpoint),
            str(checkpoint.resolve()),
        ),
    )


def resolve_cooperator_restart(restart_from: str | Path | None) -> tuple[str | None, str | None]:
    """Resolve a cooperator restart target to ``(restore_file, output_dir)``.

    ``restart_from`` may point to:
    - a specific ``checkpoint_*.pt`` file
    - an experiment directory containing ``checkpoints/``
    - a cooperator run directory containing one or more experiment subdirs

    The returned output directory is the cooperator run directory passed to
    ``env_module.train_fn(..., output_dir=...)``.
    """
    if restart_from is None:
        return None, None

    restart_path = Path(restart_from).expanduser()
    if not restart_path.is_absolute():
        restart_path = Path(to_absolute_path(str(restart_path)))
    restart_path = restart_path.resolve()

    if not restart_path.exists():
        raise FileNotFoundError(f"Restart path does not exist: {restart_path}")

    checkpoint = restart_path if restart_path.is_file() else _find_latest_checkpoint_in_dir(restart_path)
    if checkpoint.name.startswith("checkpoint_") is False or checkpoint.suffix != ".pt":
        raise ValueError(f"Restart checkpoint must be a checkpoint_*.pt file, got {checkpoint}")

    run_dir = checkpoint.parent.parent.parent.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Resolved cooperator output directory does not exist: {run_dir}")

    return str(checkpoint), str(run_dir)


def infer_self_play_results_dir_from_cooperator_run(run_dir: str | Path) -> str:
    """Infer the parent self-play run directory from a cooperator run directory."""
    run_path = Path(run_dir).resolve()
    for candidate in run_path.parents:
        seed_dirs = sorted(child for child in candidate.glob("seed_*") if child.is_dir())
        if seed_dirs:
            return str(candidate)
    raise FileNotFoundError(
        "Could not infer self-play results dir from restarted cooperator run. "
        f"No ancestor of {run_path} contains seed_*/ subdirectories."
    )


def train_cooperator_against_population(
    env_name: str,
    self_play_results_dir: str | Path,
    output_dir: str,
    train_kwargs: dict[str, Any],
) -> None:
    """Train a cooperator against the latest checkpoint of each self-play seed."""
    env_module = resolve_env_module(env_name, required_attrs=("train_fn",))
    reload_context = getattr(env_module, "get_cooperator_reload_context", None)
    seed_dirs = _enumerate_seed_dirs(Path(self_play_results_dir))
    policies = [load_selfplay_policy(sd, reload_context=reload_context) for sd in seed_dirs]
    logging.info("Loaded %d self-play policies for cooperator training", len(policies))

    adapted_policies = [SelfPlayPolicyTensorAdapter(p) for p in policies]
    run_date = extract_run_date_from_path(output_dir)
    run_name = build_cooperator_run_name(env_name=env_name, run_date=run_date)

    sanitized_kwargs = dict(train_kwargs)
    sanitized_kwargs.pop("run_name", None)
    sanitized_kwargs["prioritized"] = False

    os.makedirs(output_dir, exist_ok=True)
    env_module.train_fn(
        reward_fn=None,
        output_dir=output_dir,
        run_name=run_name,
        teammate_policy=[adapted_policies],
        **sanitized_kwargs,
    )


def train_fcp_against_population(
    env_name: str,
    self_play_results_dir: str | Path,
    output_dir: str,
    train_kwargs: dict[str, Any],
    max_checkpoints_per_seed: int | None = None,
) -> dict[str, Any]:
    """Train an FCP cooperator against checkpoints sampled across training time."""
    env_module = resolve_env_module(env_name, required_attrs=("train_fn",))
    source_dir = Path(self_play_results_dir).resolve()
    checkpoint_paths = discover_self_play_checkpoints(source_dir, max_checkpoints_per_seed=max_checkpoints_per_seed)
    logging.info("Loading %d FCP checkpoint policies onto CUDA before training.", len(checkpoint_paths))

    reload_context = getattr(env_module, "get_cooperator_reload_context", None)
    policies = load_checkpoint_policies(checkpoint_paths, reload_context=reload_context)
    logging.info(
        "Loaded %d checkpoint policies across %d seeds for FCP training",
        len(policies),
        len({p.seed_dir for p in policies}),
    )

    adapted_policies = [SelfPlayPolicyTensorAdapter(p) for p in policies]
    run_date = extract_run_date_from_path(output_dir)
    run_name = build_cooperator_run_name(env_name=env_name, run_date=run_date)

    sanitized_kwargs = dict(train_kwargs)
    sanitized_kwargs.pop("run_name", None)
    sanitized_kwargs["prioritized"] = False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "env": env_name,
        "self_play_results_dir": str(source_dir),
        "output_dir": str(output_path.resolve()),
        "n_seed_runs": len({p.seed_dir for p in policies}),
        "n_checkpoint_policies": len(policies),
        "max_checkpoints_per_seed": max_checkpoints_per_seed,
        "checkpoint_policies_device": "cuda",
        "checkpoint_paths": [str(Path(p).resolve()) for p in checkpoint_paths],
    }
    metadata_path = output_path / "fcp_population.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    metadata["metadata_path"] = str(metadata_path.resolve())

    env_module.train_fn(
        reward_fn=None,
        output_dir=str(output_path),
        run_name=run_name,
        teammate_policy=[adapted_policies],
        **sanitized_kwargs,
    )
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_self_play_results_dir(cfg: DictConfig) -> str:
    if cfg.cooperator.self_play_results_dir is not None:
        return to_absolute_path(str(cfg.cooperator.self_play_results_dir))
    return to_absolute_path(os.path.join(cfg.self_play.output_dir, cfg.self_play.run_name))


def run_self_play_cooperator(cfg: DictConfig) -> str:
    train_kwargs = OmegaConf.to_container(cfg.cooperator.train_kwargs, resolve=True)
    restart_from = cfg.cooperator.get("restart_from")
    restore_file, restart_output_dir = resolve_cooperator_restart(restart_from)
    results_dir = resolve_self_play_results_dir(cfg)

    output_dir = os.path.join(results_dir, cfg.cooperator.output_subdir, cfg.cooperator.run_name)
    if restore_file is not None:
        if cfg.cooperator.self_play_results_dir is None:
            results_dir = infer_self_play_results_dir_from_cooperator_run(restart_output_dir)
        existing_restore = train_kwargs.get("restore_file")
        if existing_restore not in (None, restore_file):
            raise ValueError(
                "Conflicting cooperator restore targets: "
                f"cooperator.restart_from resolved to {restore_file}, "
                f"but cooperator.train_kwargs.restore_file is {existing_restore}."
            )
        train_kwargs["restore_file"] = restore_file
        output_dir = restart_output_dir
        logging.info(
            "Restarting cooperator training from checkpoint=%s into output_dir=%s",
            restore_file,
            output_dir,
        )

    logging.info("Cooperator pipeline | env=%s | results=%s | output=%s", cfg.env, results_dir, output_dir)
    train_cooperator_against_population(
        env_name=cfg.env,
        self_play_results_dir=results_dir,
        output_dir=output_dir,
        train_kwargs=train_kwargs,
    )
    return output_dir


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    """Train a cooperator against an existing self-play population.

    Run from this directory:
        python cooperator_training.py env=lbf
        python cooperator_training.py env=lbf cooperator.self_play_results_dir=/path/to/sp_run
        python cooperator_training.py env=lbf cooperator.restart_from=/path/to/cooperator_run
        python cooperator_training.py env=lbf cooperator.restart_from=/path/to/checkpoints/checkpoint_804000.pt

    Key config fields (cfg/config.yaml):
        env                               Environment name (lbf, football, smacv2, ...)
        cooperator.self_play_results_dir  Path to an existing self-play run (seed_*/ dirs).
                                          If unset, uses <self_play.output_dir>/<self_play.run_name>.
        cooperator.restart_from           Optional checkpoint file, experiment dir, or cooperator run dir
                                          to resume in place from the latest checkpoint.
        cooperator.output_subdir          Subdir under the run directory for the cooperator.
        cooperator.run_name               Name of this cooperator training run.
        cooperator.train_kwargs           Extra kwargs forwarded to env_module.train_fn.
    """
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    run_self_play_cooperator(cfg)


if __name__ == "__main__":
    main()
