import os
import time
from collections.abc import Callable

import torch



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
