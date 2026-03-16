import importlib
import logging
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from pqdm.processes import pqdm
from tensordict import TensorDict

from popgen.utils.run_names import (
    build_cooperator_run_name,
    extract_run_date_from_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def resolve_env_module(env_name: str, required_attrs: Sequence[str] = ()) -> Any:
    env_module = importlib.import_module(f"popgen.envs.{env_name}.utils")
    for attr in required_attrs:
        if not hasattr(env_module, attr):
            raise AttributeError(f"popgen.envs.{env_name}.utils must define {attr}")
    return env_module


def resolve_seeds(seeds: Sequence[int] | None, n_seeds: int) -> list[int]:
    if seeds is None:
        resolved = list(range(n_seeds))
    else:
        resolved = [int(seed) for seed in seeds]
    if not resolved:
        raise ValueError("No seeds configured. Provide seeds or set n_seeds > 0.")
    return resolved


def _build_task(env_name: str, rl_cfg_dir: Path):
    if env_name in {"point_mass_rendezvous", "football"}:
        return VmasTask[env_name.upper()].get_from_yaml(path=str(rl_cfg_dir / f"{env_name}.yaml"))
    if env_name == "lbf":
        from benchmarl.environments.lbf.common import LbfTask

        return LbfTask.LBF.get_from_yaml(path=str(rl_cfg_dir / f"{env_name}.yaml"))
    raise ValueError(f"Unsupported environment: {env_name}")


def _train_single_seed(args: tuple[int, str, str, dict[str, Any]]):
    seed, seed_output_dir, env_name, train_kwargs = args
    env_module = resolve_env_module(env_name, required_attrs=("RL_CFG_DIR",))
    rl_cfg_dir: Path = env_module.RL_CFG_DIR

    os.makedirs(seed_output_dir, exist_ok=True)
    task = _build_task(env_name, rl_cfg_dir)

    mappo_path = rl_cfg_dir / "mappo.yaml"
    algorithm_config = MappoConfig.get_from_yaml(path=str(mappo_path)) if mappo_path.exists() else MappoConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml(path=str(rl_cfg_dir / "base_experiment.yaml"))
    model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / "mlp.yaml"))
    critic_model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / "mlp.yaml"))

    for key, value in train_kwargs.items():
        setattr(experiment_config, key, value)

    experiment_config.save_folder = seed_output_dir
    experiment_config.wandb_extra_kwargs = {"name": f"{env_name}_selfplay_seed_{seed}", "tags": [env_name, "self_play"]}

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
    )
    experiment.run()
    logging.info(f"[seed={seed}] Training complete -> {seed_output_dir}")
    return seed_output_dir


def run_self_play_population(
    env_name: str,
    output_dir: str,
    seeds: Sequence[int],
    n_workers: int,
    train_kwargs: dict[str, Any],
) -> list[str]:
    resolve_env_module(env_name, required_attrs=("RL_CFG_DIR",))
    os.makedirs(output_dir, exist_ok=True)

    jobs = [
        (
            int(seed),
            os.path.join(output_dir, f"seed_{int(seed)}"),
            env_name,
            dict(train_kwargs),
        )
        for seed in seeds
    ]
    results = pqdm(jobs, _train_single_seed, n_jobs=n_workers, desc="Self-play training")
    failures = [result for result in results if isinstance(result, Exception)]
    if failures:
        raise RuntimeError(f"{len(failures)} self-play seeds failed. First error: {failures[0]}")
    return [str(result) for result in results]


def _get_n_agents(env) -> int:
    if hasattr(env, "n_agents"):
        return int(env.n_agents)
    if hasattr(env, "num_agents"):
        return int(env.num_agents)
    for group in ("agents", "ego"):
        try:
            return env.observation_spec[group, "observation"].shape[-2]
        except (KeyError, IndexError, AttributeError):
            pass
    raise RuntimeError(f"Could not determine n_agents from env {type(env).__name__}")


class SelfPlayPolicyWrapper(nn.Module):
    """Adapts a jointly-trained N-agent policy for single-agent teammate calls."""

    def __init__(self, policy, trained_group: str = "agents", n_agents_trained: int = 2):
        super().__init__()
        self.policy = policy
        self.trained_group = trained_group
        self.n_agents_trained = n_agents_trained

    def forward(self, td):
        call_group = next(iter(td.keys()))
        input_device = td.device

        try:
            policy_device = next(self.policy.parameters()).device
        except StopIteration:
            policy_device = input_device

        single_obs = td[call_group]["observation"]
        n_batch = single_obs.shape[0]
        obs_dim = single_obs.shape[-1]

        padded_obs = single_obs.expand(n_batch, self.n_agents_trained, obs_dim).contiguous().to(policy_device)
        remapped_in = TensorDict(
            {
                self.trained_group: TensorDict(
                    {"observation": padded_obs},
                    batch_size=[n_batch, self.n_agents_trained],
                    device=policy_device,
                )
            },
            batch_size=[n_batch],
            device=policy_device,
        )

        with torch.no_grad():
            out = self.policy(remapped_in)

        out_single = out[self.trained_group][:, :1].to(input_device)
        return TensorDict({call_group: out_single}, batch_size=[n_batch], device=input_device)


class _StubCallable:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Stub callable: original dynamic module not available")


def _register_dynamic_stubs():
    if "popgen._dynamic" not in sys.modules:
        pkg = types.ModuleType("popgen._dynamic")
        pkg.__package__ = "popgen._dynamic"
        pkg.__path__ = []
        sys.modules["popgen._dynamic"] = pkg

    class _StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("popgen._dynamic."):
                if module not in sys.modules:
                    stub = types.ModuleType(module)
                    stub.__package__ = "popgen._dynamic"
                    sys.modules[module] = stub
                mod = sys.modules[module]
                if not hasattr(mod, name):
                    setattr(mod, name, _StubCallable())
                return getattr(mod, name)
            return super().find_class(module, name)

    return _StubUnpickler


def load_latest_policy_and_env(
    save_dir: str | Path,
    allow_dynamic_stubs: bool = False,
    move_policy_to_cpu: bool = False,
):
    save_dir_path = Path(save_dir)
    checkpoints = sorted(
        list(save_dir_path.rglob("checkpoints/checkpoint_*.pt")),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {save_dir_path}")

    checkpoint = checkpoints[-1]
    experiment_folder = checkpoint.parent.parent.resolve()

    original_pickle_load = pickle.load
    if allow_dynamic_stubs:
        stub_unpickler = _register_dynamic_stubs()

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
            },
        )
    finally:
        pickle.load = original_pickle_load

    policy_group = "ego" if "ego" in experiment.group_policies else "agents"
    policy = experiment.group_policies[policy_group]
    if move_policy_to_cpu:
        policy = policy.cpu()
    env = experiment.test_env
    return policy, env, policy_group


def close_env_safely(env) -> None:
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def load_self_play_policies(
    self_play_results_dir: str | Path,
    allow_dynamic_stubs: bool = False,
    move_policy_to_cpu: bool = False,
) -> list[SelfPlayPolicyWrapper]:
    results_dir = Path(self_play_results_dir)
    seed_dirs = sorted(results_dir.glob("seed_*/"), key=lambda path: path.name)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_*/ subdirectories found in {results_dir}")

    policies: list[SelfPlayPolicyWrapper] = []
    for seed_dir in seed_dirs:
        logging.info(f"Loading self-play policy from {seed_dir}")
        raw_policy, env, trained_group = load_latest_policy_and_env(
            seed_dir,
            allow_dynamic_stubs=allow_dynamic_stubs,
            move_policy_to_cpu=move_policy_to_cpu,
        )
        n_agents = _get_n_agents(env)
        close_env_safely(env)
        logging.info(f"  n_agents_trained={n_agents}")
        policies.append(
            SelfPlayPolicyWrapper(
                raw_policy,
                trained_group=trained_group,
                n_agents_trained=n_agents,
            )
        )
    return policies


def train_cooperator_against_population(
    env_name: str,
    self_play_results_dir: str | Path,
    output_dir: str,
    train_kwargs: dict[str, Any],
) -> None:
    env_module = resolve_env_module(env_name, required_attrs=("train_fn",))
    policies = load_self_play_policies(self_play_results_dir)
    logging.info(f"Loaded {len(policies)} self-play policies for cooperator training")
    run_date = extract_run_date_from_path(output_dir)
    run_name = build_cooperator_run_name(env_name=env_name, run_date=run_date)
    sanitized_train_kwargs = dict(train_kwargs)
    if "run_name" in sanitized_train_kwargs:
        logging.warning("Ignoring train_kwargs['run_name']; using canonical computed run_name.")
        sanitized_train_kwargs.pop("run_name", None)

    os.makedirs(output_dir, exist_ok=True)
    env_module.train_fn(
        reward_fn=None,
        output_dir=output_dir,
        run_name=run_name,
        teammate_policy=[policies],
        **sanitized_train_kwargs,
    )


def find_cooperator_dir(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    cooperator_root = run_path / "cooperator"
    if not cooperator_root.exists():
        raise FileNotFoundError(f"No cooperator/ directory found under {run_path}")

    subdirs = sorted(path for path in cooperator_root.iterdir() if path.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {cooperator_root}")

    latest = subdirs[-1]
    logging.info(f"Auto-detected cooperator dir: {latest}")
    return latest
