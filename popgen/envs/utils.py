import os
import time
from collections.abc import Callable

import torch


def parse_single_scripted_policy(
    scripted_policy: dict[int, Callable],
) -> tuple[int, Callable]:
    """Validate and unpack a single scripted teammate mapping."""
    if not isinstance(scripted_policy, dict):
        raise TypeError("scripted_policy must be a dict[int, callable].")
    if len(scripted_policy) != 1:
        raise ValueError(
            f"Exactly one scripted agent is supported, got {len(scripted_policy)} entries."
        )

    teammate_idx, scripted_fn = next(iter(scripted_policy.items()))
    if not isinstance(teammate_idx, int):
        raise TypeError(
            f"Scripted agent index must be int, got {type(teammate_idx).__name__}."
        )
    if teammate_idx < 0:
        raise ValueError(f"Scripted agent index must be non-negative, got {teammate_idx}.")
    if not callable(scripted_fn):
        raise TypeError("scripted_policy value must be callable.")
    return teammate_idx, scripted_fn


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
