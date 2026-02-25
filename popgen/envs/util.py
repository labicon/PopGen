import os
import time
from collections.abc import Callable
from typing import Any

import torch
from popgen.agents.scripted_partner import MultiPolicyWrapper


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


def compose_rollout_policy(policy_or_policies: Any):
    """Normalize rollout input to a single policy module.

    Accepts:
        - single policy module
        - [ego_policy]
        - [ego_policy, teammate_policy] -> wrapped into MultiPolicyWrapper
    """
    if isinstance(policy_or_policies, (list, tuple)):
        if len(policy_or_policies) == 1:
            policy = policy_or_policies[0]
        elif len(policy_or_policies) == 2:
            policy = MultiPolicyWrapper(
                ego=(0, policy_or_policies[0]),
                teammate=(1, policy_or_policies[1]),
            )
        else:
            raise ValueError(
                f"Expected 1 or 2 policies for rollout, got {len(policy_or_policies)}."
            )
    else:
        policy = policy_or_policies

    if not hasattr(policy, "eval"):
        raise TypeError(
            f"Policy must be nn.Module-like with eval(), got {type(policy).__name__}."
        )

    return policy


def apply_rollout_seed(env, seed: int | None = None) -> None:
    """Apply a seed to the environment for reproducible rollouts.

    If seed is None, derives one from time and PID so parallel workers with
    overlapping wall-clock milliseconds still get different seeds.
    """
    if seed is None:
        seed = (int(time.time() * 1000) + os.getpid()) % 100000
    torch.manual_seed(seed)
    if hasattr(env, "set_seed"):
        env.set_seed(seed)
