"""TensorDict helpers for policy routing."""

from __future__ import annotations

from tensordict import TensorDictBase


def find_observation_group(tensordict: TensorDictBase) -> str:
    """Return the single observation group name from a TensorDict."""
    nested_keys = tensordict.keys(include_nested=True, leaves_only=True)
    observation_keys = [key for key in nested_keys if isinstance(key, tuple) and len(key) == 2 and key[1] == "observation"]
    groups = list({key[0] for key in observation_keys})
    if len(groups) == 1 and groups[0] == "agents":
        return "agents"
    if len(groups) != 1:
        print(f"Received tensordict: {tensordict}")
        raise KeyError(f"Expected exactly one observation group. Found: {observation_keys}")
    return groups[0]
