import importlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

# Registry: yaml stem → BenchMARL config class.
# Order defines priority when multiple matching files exist in a directory.
_ALGORITHM_REGISTRY: dict[str, str] = {
    "ippo": "benchmarl.algorithms.IppoConfig",
    "mappo": "benchmarl.algorithms.MappoConfig",
    "iql": "benchmarl.algorithms.IqlConfig",
    "iddpg": "benchmarl.algorithms.IddpgConfig",
    "qmix": "benchmarl.algorithms.QmixConfig",
    "vdn": "benchmarl.algorithms.VdnConfig",
    "ppo": "benchmarl.algorithms.PpoConfig",
    "maddpg": "benchmarl.algorithms.MaddpgConfig",
    "isac": "benchmarl.algorithms.IsacConfig",
    "masac": "benchmarl.algorithms.MasacConfig",
}

_MODEL_REGISTRY: dict[str, str] = {
    "gru": "benchmarl.models.GruConfig",
    "lstm": "benchmarl.models.LstmConfig",
    "mlp": "benchmarl.models.MlpConfig",
    "cnn": "benchmarl.models.CnnConfig",
    "gnn": "benchmarl.models.GnnConfig",
    "deepsets": "benchmarl.models.DeepsetsConfig",
}


def _import_dotted(dotted: str):
    module_path, _, attr = dotted.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def resolve_cfg_dir(rl_cfg_dir: Path, teammate_policy) -> Path:
    """Return the mode-specific subdirectory of *rl_cfg_dir*.

    - ``teammate_policy is None``   → ``self_play/``
    - flat list                     → ``rl_reward/``
    - list of lists                 → ``population_play/``

    The algorithm used within each subdirectory is determined by whichever
    ``{algo}.yaml`` file is present (scanned in ``_ALGORITHM_REGISTRY`` order).

    Falls back to *rl_cfg_dir* itself if the subdir does not exist.
    """
    if teammate_policy is None:
        subdir = "self_play"
    else:
        subdir = "population_play" if isinstance(teammate_policy[0], list) else "rl_reward"
    candidate = rl_cfg_dir / subdir
    return candidate if candidate.exists() else rl_cfg_dir


def load_algorithm_config(cfg_dir: Path):
    """Load the algorithm config from *cfg_dir* by scanning for known yaml filenames.

    The first registry entry whose ``{name}.yaml`` file exists in *cfg_dir* is
    used.  Raises ``FileNotFoundError`` if none match.
    """
    for name, dotted_cls in _ALGORITHM_REGISTRY.items():
        path = cfg_dir / f"{name}.yaml"
        if path.exists():
            cls = _import_dotted(dotted_cls)
            return cls.get_from_yaml(path=str(path))
    known = ", ".join(f"{n}.yaml" for n in _ALGORITHM_REGISTRY)
    raise FileNotFoundError(f"No algorithm config found in '{cfg_dir}'. Expected one of: {known}.")


def _load_sequence_model_config(path: Path):
    """Load a SequenceModelConfig from a flat yaml listing layer names and intermediate_sizes."""
    import yaml
    from benchmarl.models import SequenceModelConfig

    with open(path) as f:
        cfg = yaml.safe_load(f)

    layer_names = cfg["layers"]
    intermediate_sizes = cfg["intermediate_sizes"]

    model_configs = []
    for name in layer_names:
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown layer '{name}' in sequence config '{path}'. Known: {list(_MODEL_REGISTRY)}")
        layer_path = path.parent / f"{name}.yaml"
        if not layer_path.exists():
            raise FileNotFoundError(f"Layer config '{layer_path}' not found.")
        cls = _import_dotted(_MODEL_REGISTRY[name])
        model_configs.append(cls.get_from_yaml(path=str(layer_path)))

    return SequenceModelConfig(model_configs=model_configs, intermediate_sizes=intermediate_sizes)


def load_model_config(cfg_dir: Path, *, prefix: str = "", name: str | None = None):
    """Load a model config from *cfg_dir* by scanning for known yaml filenames.

    If *prefix* is given (e.g. ``"critic_"``), looks for ``critic_{arch}.yaml``
    files.  If *name* is given, load that specific architecture directly without
    scanning.  Raises ``FileNotFoundError`` if none match.
    """
    if name is not None:
        if name == "sequence":
            path = cfg_dir / f"{prefix}sequence.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Model config '{path}' not found.")
            return _load_sequence_model_config(path)
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model config name '{name}'. Known: {list(_MODEL_REGISTRY)}")
        path = cfg_dir / f"{prefix}{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Model config '{path}' not found.")
        cls = _import_dotted(_MODEL_REGISTRY[name])
        return cls.get_from_yaml(path=str(path))

    # SequenceModelConfig.get_from_yaml() raises NotImplementedError in BenchMARL,
    # so handle sequence configs separately before the generic registry scan.
    seq_path = cfg_dir / f"{prefix}sequence.yaml"
    if seq_path.exists():
        return _load_sequence_model_config(seq_path)

    for arch, dotted_cls in _MODEL_REGISTRY.items():
        path = cfg_dir / f"{prefix}{arch}.yaml"
        if path.exists():
            cls = _import_dotted(dotted_cls)
            return cls.get_from_yaml(path=str(path))
    known = ", ".join(f"{prefix}{n}.yaml" for n in _MODEL_REGISTRY)
    raise FileNotFoundError(f"No model config found in '{cfg_dir}' with prefix '{prefix!r}'. Expected one of: {known}.")


def compose_rollout_policy(policy_or_policies: Any):
    """Normalize rollout input to a single policy module.

    Accepts:
        - single policy module
        - [ego_policy]
        - [ego_policy, teammate_policy] -> wrapped into RLEgoTeammateRouter
          when ego was trained in split-group mode.
    """
    if isinstance(policy_or_policies, (list, tuple)):
        if len(policy_or_policies) == 1:
            policy = policy_or_policies[0]
        elif len(policy_or_policies) == 2:
            ego, teammate = policy_or_policies[0], policy_or_policies[1]
            ego_group = getattr(ego, "_benchmarl_ego_group", "agents")
            if ego_group == "agents":
                raise NotImplementedError(
                    "Single-group rollout composition has been removed with "
                    "policy_wrappers.py. Reimplement single-group support before use."
                )
            else:
                from popgen.agents.router import RLEgoTeammateRouter

                policy = RLEgoTeammateRouter(ego_policy=ego, teammate_policies=[teammate])
        else:
            raise ValueError(f"Expected 1 or 2 policies for rollout, got {len(policy_or_policies)}.")
    else:
        policy = policy_or_policies

    if not hasattr(policy, "eval"):
        raise TypeError(f"Policy must be nn.Module-like with eval(), got {type(policy).__name__}.")

    return policy


def _normalize_nested_key(key: Any) -> tuple[str, ...]:
    if isinstance(key, tuple):
        return tuple(str(part) for part in key)
    return (str(key),)


def _iter_reward_leaf_paths(step_td: Any) -> list[tuple[str, ...]]:
    try:
        keys = step_td.keys(include_nested=True, leaves_only=True)
    except TypeError:
        keys = step_td.keys(include_nested=True)

    reward_paths: list[tuple[str, ...]] = []
    for key in keys:
        path = _normalize_nested_key(key)
        if path and path[-1] == "reward":
            reward_paths.append(path)
    return reward_paths


def _select_reward_paths_by_group(reward_paths: list[tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    # Prefer ("next", group, "reward") over (group, "reward") when both exist.
    selected: dict[str, tuple[int, tuple[str, ...]]] = {}
    for path in reward_paths:
        if path[0] == "next" and len(path) >= 3:
            group = path[1]
            priority = 0
        elif len(path) >= 2:
            group = path[0]
            priority = 1
        else:
            group = "root"
            priority = 2

        previous = selected.get(group)
        if previous is None or priority < previous[0]:
            selected[group] = (priority, path)

    return {group: entry[1] for group, entry in selected.items()}


def _reward_value_mean(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if hasattr(value, "float") and callable(value.float):
            mean_value = value.float().mean()
        elif hasattr(value, "mean") and callable(value.mean):
            mean_value = value.mean()
        else:
            return float(value)

        if hasattr(mean_value, "item") and callable(mean_value.item):
            return float(mean_value.item())
        return float(mean_value)
    except Exception:
        return None


def extract_reward_rollout_info(rollout_td: Any) -> dict[str, Any]:
    """Aggregate rollout rewards by group from a rollout TensorDict-like object.

    Reward extraction scans nested leaves that end with "reward". If both
    ``("next", group, "reward")`` and ``(group, "reward")`` are present for a
    group at one step, the ``"next"`` value is used.
    """
    reward_totals_by_group: dict[str, float] = {}
    reward_steps = 0

    rollout_batch_size = getattr(rollout_td, "batch_size", ())

    # Vectorized envs (e.g., VMAS) return (n_envs, T) rollouts where dim 0 is
    # n_envs and dim 1 is time. Normalize to (T,) by selecting env 0.
    if len(rollout_batch_size) >= 2:
        rollout_td = rollout_td[0]
        rollout_batch_size = rollout_td.batch_size

    n_steps = int(rollout_batch_size[0]) if rollout_batch_size else 0

    for step_idx in range(n_steps):
        step_td = rollout_td[step_idx]
        reward_paths = _iter_reward_leaf_paths(step_td)
        selected_paths = _select_reward_paths_by_group(reward_paths)
        if not selected_paths:
            continue

        step_has_reward = False
        for group, path in selected_paths.items():
            try:
                reward_value = _reward_value_mean(step_td[path])
            except Exception:
                reward_value = None

            if reward_value is None:
                continue

            reward_totals_by_group[group] = reward_totals_by_group.get(group, 0.0) + reward_value
            step_has_reward = True

        if step_has_reward:
            reward_steps += 1

    reward_total_all_groups = float(sum(reward_totals_by_group.values()))
    return {
        "reward_totals_by_group": reward_totals_by_group,
        "reward_total_all_groups": reward_total_all_groups,
        "reward_steps": reward_steps,
    }



def apply_train_kwargs(kwargs: dict, *cfg_objects) -> None:
    """Apply remaining train kwargs to BenchMARL config objects.

    Iterates over *kwargs* and sets each key on the first config object that
    has a matching attribute.  Unknown keys are logged as warnings.  Mirrors
    the pattern used in ``train_lbf_env``.
    """
    for key, val in kwargs.items():
        applied = False
        for cfg_obj in cfg_objects:
            if hasattr(cfg_obj, key):
                setattr(cfg_obj, key, val)
                applied = True
                break
        if not applied:
            logging.warning(f"train_fn: unknown train_kwarg '{key}' — ignored.")


def resolve_rollout_seed(seed: int | None = None) -> int:
    """Resolve the effective rollout seed.

    If ``seed`` is ``None``, derives one from time and PID so parallel workers
    with overlapping wall-clock milliseconds still get different seeds.
    """
    if seed is None:
        return (int(time.time() * 1000) + os.getpid()) % 100000
    return int(seed)


def apply_rollout_seed(env, seed: int | None = None) -> None:
    """Apply a seed to the environment for reproducible rollouts.

    If seed is None, derives one from time and PID so parallel workers with
    overlapping wall-clock milliseconds still get different seeds.
    """
    seed = resolve_rollout_seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, "set_seed"):
        env.set_seed(seed)
