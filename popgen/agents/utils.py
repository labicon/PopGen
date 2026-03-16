"""Shared helpers for agent I/O normalization."""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase


def find_observation_group(tensordict: TensorDictBase) -> str:
    """Return the single observation group name from a TensorDict.

    Fast-paths on the conventional ``"agents"`` group; otherwise scans all
    nested leaf keys for tuples of the form ``(group, "observation")`` and
    asserts exactly one group is present.

    Args:
        tensordict: The input TensorDict to inspect.

    Returns:
        The group name (e.g. ``"agents"``, ``"ego"``).

    Raises:
        KeyError: If zero or more than one observation group is found.
    """
    nested_keys = tensordict.keys(include_nested=True, leaves_only=True)
    if ("agents", "observation") in nested_keys:
        return "agents"
    if ("ego", "observation") in nested_keys:
        return "ego"
    observation_keys = [key for key in nested_keys if isinstance(key, tuple) and len(key) == 2 and key[1] == "observation"]
    groups = list({key[0] for key in observation_keys})
    if len(groups) != 1:
        print(f"Received tensordict: {tensordict}")
        raise KeyError(f"Expected exactly one observation group. Found: {observation_keys}")
    return groups[0]


def extract_observation_and_action_key(obs):
    """Resolve the observation tensor and corresponding action key from *obs*.

    Normalizes the three input formats used across agent implementations:

    - ``torch.Tensor`` – returned as-is with ``action_key=None``.
    - ``TensorDictBase`` with a flat ``"observation"`` key – returns
      ``(obs["observation"], "action")``.
    - ``TensorDictBase`` with a grouped layout – delegates to
      :func:`find_observation_group` and returns
      ``(obs[group, "observation"], (group, "action"))``.
    - ``dict`` with an ``"observation"`` key – returns
      ``(obs["observation"], None)``.

    Args:
        obs: Raw policy input in one of the formats described above.

    Returns:
        A ``(observation_tensor, action_key)`` tuple where *action_key* is
        the TensorDict key to use when writing the output action, or ``None``
        for plain-tensor inputs.

    Raises:
        KeyError: If a dict input lacks the ``"observation"`` key.
        TypeError: If *obs* is none of the supported types.
    """
    if isinstance(obs, torch.Tensor):
        return obs, None

    if isinstance(obs, TensorDictBase):
        if "observation" in obs.keys():
            return obs["observation"], "action"

        group = find_observation_group(obs)
        return obs[group, "observation"], (group, "action")

    if isinstance(obs, dict):
        if "observation" not in obs:
            raise KeyError('Expected key "observation" in dict input.')
        return obs["observation"], None

    raise TypeError(f"Unsupported input type: {type(obs).__name__}")


def format_action_output(obs, action_key, action_tensor: torch.Tensor):
    """Package *action_tensor* into the same container type as *obs*.

    If *obs* is a ``TensorDictBase``, creates an empty TensorDict of the same
    batch size and sets *action_key* to *action_tensor*.  For plain-tensor
    inputs the action tensor is returned directly.

    Args:
        obs: The original policy input (used only to determine the output type).
        action_key: Key under which to store the action in the output TensorDict;
            may be a string or nested tuple (e.g. ``(group, "action")``).
        action_tensor: The computed action to package.

    Returns:
        A TensorDict with *action_key* set, or *action_tensor* directly.
    """
    if isinstance(obs, TensorDictBase):
        # Return a minimal output container to avoid carrying over nested
        # rollout internals (e.g. "next" trees) from view TensorDicts.
        out = TensorDict({}, batch_size=obs.batch_size, device=action_tensor.device)
        out.set(action_key, action_tensor)
        return out
    return action_tensor


def flatten_obs(obs_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Normalize a batched observation tensor to 2-D.

    Returns:
        flat_obs: shape ``(N, obs_dim)``
        leading_shape: original leading dimensions, or ``torch.Size([])`` for 1-D input
    """
    if obs_tensor.dim() == 1:
        return obs_tensor.unsqueeze(0), torch.Size([])
    leading_shape = obs_tensor.shape[:-1]
    return obs_tensor.reshape(-1, obs_tensor.shape[-1]), leading_shape
