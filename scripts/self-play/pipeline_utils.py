import importlib
import logging
import random
import re
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarl.environments import VmasTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

_CHECKPOINT_STEP_RE = re.compile(r"^checkpoint_(\d+)$")


def resolve_env_module(env_name: str, required_attrs: Sequence[str] = ()) -> Any:
    env_module = importlib.import_module(f"popgen.envs.{env_name}.utils")
    for attr in required_attrs:
        if not hasattr(env_module, attr):
            raise AttributeError(f"popgen.envs.{env_name}.utils must define {attr}")
    return env_module


def resolve_seeds(
    seeds: Sequence[int] | None,
    n_seeds: int,
    population_seed: int | None = None,
) -> list[int]:
    if seeds is None:
        count = int(n_seeds)
        if count <= 0:
            raise ValueError("No seeds configured. Provide seeds or set n_seeds > 0.")
        if population_seed is None:
            resolved = list(range(count))
        else:
            rng = random.Random(int(population_seed))
            resolved = rng.sample(range(2**31), count)
    else:
        resolved = [int(seed) for seed in seeds]
    if not resolved:
        raise ValueError("No seeds configured. Provide seeds or set n_seeds > 0.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Self-play seeds must be unique. Got duplicates in {resolved}.")
    return resolved


def _build_task(env_name: str, rl_cfg_dir: Path):
    if env_name in {"point_mass_rendezvous", "football"}:
        task = VmasTask[env_name.upper()].get_from_yaml(path=str(rl_cfg_dir / f"{env_name}.yaml"))
        env_module = resolve_env_module(env_name)
        if hasattr(env_module, "SELFPLAY_SCENARIO_KWARGS"):
            task.config.update(env_module.SELFPLAY_SCENARIO_KWARGS)
        return task
    if env_name == "pp":
        task = VmasTask.SIMPLE_TAG.get_from_yaml(path=str(rl_cfg_dir / "simple_tag.yaml"))
        num_adversaries = int(task.config.get("num_adversaries", 2))
        adversary_names = [f"adversary_{i}" for i in range(num_adversaries)]
        task.config["group_map"] = {"adversary": adversary_names}
        return task
    if env_name == "pp_stag":
        task = VmasTask.SIMPLE_TAG.get_from_yaml(path=str(rl_cfg_dir / "simple_tag.yaml"))
        num_adversaries = int(task.config.get("num_adversaries", 2))
        adversary_names = [f"adversary_{i}" for i in range(num_adversaries)]
        task.config["group_map"] = {"adversary": adversary_names}
        return task
    if env_name == "lbf":
        from benchmarl.environments.lbf.common import LbfTask

        return LbfTask.LBF.get_from_yaml(path=str(rl_cfg_dir / f"{env_name}.yaml"))
    if env_name == "smacv2":
        from benchmarl.environments import Smacv2Task

        return Smacv2Task.TERRAN_2_VS_4.get_from_yaml(path=str(rl_cfg_dir / "terran_2_vs_4.yaml"))
    if env_name == "overcooked":
        from popgen.envs.overcooked.overcooked_benchmarl import OvercookedTask

        task = OvercookedTask.OVERCOOKED.get_from_yaml(path=str(rl_cfg_dir / "overcooked.yaml"))
        task.config["group_map"] = {"agent": ["agent_0", "agent_1"]}
        return task
    raise ValueError(f"Unsupported environment: {env_name}")


@contextmanager
def _noop_context():
    yield


@contextmanager
def _pp_semdiv_context():
    """Patch vmas.make_env to use the semdiv scenario during pp self-play training."""
    import vmas

    from popgen.envs.pp import semdiv_simple_tag

    original_make_env = vmas.make_env

    def _patched(scenario, *args, **kwargs):
        if scenario == "simple_tag":
            return original_make_env(semdiv_simple_tag.Scenario(), *args, **kwargs)
        return original_make_env(scenario, *args, **kwargs)

    vmas.make_env = _patched
    try:
        yield
    finally:
        vmas.make_env = original_make_env


@contextmanager
def _pp_stag_semdiv_context():
    """Patch vmas.make_env to use the semdiv scenario during pp_stag self-play training."""
    import vmas

    from popgen.envs.pp_stag import semdiv_simple_tag

    original_make_env = vmas.make_env

    def _patched(scenario, *args, **kwargs):
        if scenario == "simple_tag":
            return original_make_env(semdiv_simple_tag.Scenario(), *args, **kwargs)
        return original_make_env(scenario, *args, **kwargs)

    vmas.make_env = _patched
    try:
        yield
    finally:
        vmas.make_env = original_make_env


def _checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_STEP_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Checkpoint path does not match checkpoint_<step>.pt naming: {path}")
    return int(match.group(1))


def _enumerate_seed_dirs(results_dir: Path) -> list[Path]:
    seed_dirs = sorted(results_dir.glob("seed_*/"), key=lambda path: path.name)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_*/ subdirectories found in {results_dir}")
    return seed_dirs
