"""Policy router for rollout-time multi-agent routing."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase


class MultiPolicyRouter(nn.Module):
    """Rollout router that dispatches grouped observations to per-group policies.

    Each key in ``policies`` names a group expected in the input TensorDict.
    The conventional groups used across this repository are ``"ego"`` (the
    agent being trained or evaluated) and ``"teammate"`` (partner agents),
    but any group name is accepted.

    The corresponding value determines routing behaviour:

    * **RL group** – a single :class:`TensorDictModuleBase` that receives the
      full input TensorDict and must write ``(group, "action")``.
    * **Tensor group** – a ``list[nn.Module]`` with one entry per agent.
      Each module receives ``(batch, obs_dim)`` and returns ``(batch,)``
      (discrete) or ``(batch, action_dim)`` (continuous).

    Groups present in the TensorDict but absent from ``policies`` are ignored.
    Groups present in ``policies`` but absent from the TensorDict are skipped.
    """

    def __init__(self, policies: dict[str, TensorDictModuleBase | list[nn.Module]]) -> None:
        super().__init__()
        if not policies:
            raise ValueError("policies must be a non-empty dict.")

        self._rl_groups: set[str] = set()
        for name, value in policies.items():
            if isinstance(value, TensorDictModuleBase):
                self._rl_groups.add(name)
                # Register as a named sub-module so parameters() and to() work.
                self.add_module(f"_rl_{name}", value)
            elif isinstance(value, list):
                if not value:
                    raise ValueError(f"Policy list for group {name!r} must be non-empty.")
                for i, m in enumerate(value):
                    if not isinstance(m, nn.Module):
                        raise TypeError(f"Expected nn.Module for group {name!r} index {i}, got {type(m).__name__}.")
                self.add_module(f"_tensor_{name}", nn.ModuleList(value))
            else:
                raise TypeError(f"Expected TensorDictModuleBase or list[nn.Module] for group {name!r}, got {type(value).__name__}.")

        # Store a plain dict for fast access; modules are already registered above.
        self._policies: dict[str, TensorDictModuleBase | list[nn.Module]] = dict(policies)

    # ------------------------------------------------------------------
    # Group resolution
    # ------------------------------------------------------------------

    def _resolve_active_groups(self, input_td: TensorDictBase) -> list[str]:
        """Return groups present in both the TensorDict and ``self._policies``."""
        nested_keys = set(input_td.keys(include_nested=True))
        observation_groups = [key for key in input_td.keys() if isinstance(key, str) and (key, "observation") in nested_keys]
        return [g for g in observation_groups if g in self._policies]

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_policy_device(policy: nn.Module) -> torch.device:
        try:
            return next(policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _apply_action_mask(action: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        """Replace invalid discrete actions with the first valid action per batch slot."""
        action_idx = action.long()
        selected_valid = agent_mask.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        first_valid = agent_mask.float().argmax(dim=1).to(action.dtype)
        return torch.where(selected_valid, action, first_valid)

    # ------------------------------------------------------------------
    # Tensor (per-agent) group routing
    # ------------------------------------------------------------------

    def _run_tensor_group(
        self,
        input_td: TensorDictBase,
        group: str,
        policies: list[nn.Module],
    ) -> torch.Tensor:
        # --- Extract observation, normalise to 3-D (batch, n_agents, obs_dim) ---
        group_td = input_td[group]
        obs = group_td["observation"]
        squeeze_batch = obs.dim() == 2
        if squeeze_batch:
            obs = obs.unsqueeze(0)
        batch_size, n_agents, _ = obs.shape

        if len(policies) != n_agents:
            raise ValueError(f"Group {group!r} has {n_agents} agents but {len(policies)} policies were provided.")

        obs_device = obs.device

        # --- Extract action mask, normalise to 3-D ---
        action_mask = None
        nested_keys = set(input_td.keys(include_nested=True))
        if (group, "action_mask") in nested_keys:
            raw_mask = input_td[group, "action_mask"]
            action_mask = raw_mask.unsqueeze(0) if raw_mask.dim() == 2 else raw_mask
            if action_mask.shape[0] == 1 and batch_size > 1:
                action_mask = action_mask.expand(batch_size, -1, -1)

        # --- Run each per-agent policy ---
        actions: list[torch.Tensor] = []
        for i, policy in enumerate(policies):
            agent_obs = obs[:, i, :].to(self._get_policy_device(policy))
            action = policy(agent_obs).to(obs_device)
            if not isinstance(action, torch.Tensor):
                raise TypeError(f"Policy index {i} in group {group!r} must return a torch.Tensor, got {type(action).__name__}.")
            if action.dim() not in (1, 2):
                raise ValueError(f"Policy index {i} in group {group!r} must return shape (batch,) or (batch, action_dim), got {tuple(action.shape)}.")
            if action.shape[0] != batch_size:
                raise ValueError(f"Policy index {i} in group {group!r} returned batch size {action.shape[0]}, expected {batch_size}.")
            actions.append(action)

        # --- Validate action consistency across agents ---
        ranks = {a.dim() for a in actions}
        if len(ranks) > 1:
            raise ValueError(f"Mixed action ranks in group {group!r}: policies returned ranks {sorted(ranks)}.")
        if actions[0].dim() == 2:
            dims = {a.shape[1] for a in actions}
            if len(dims) > 1:
                raise ValueError(f"Inconsistent continuous action dimensions in group {group!r}: {sorted(dims)}.")

        # --- Apply action masks for discrete actions ---
        if action_mask is not None:
            actions = [self._apply_action_mask(a, action_mask[:, i, :].to(obs_device)) if a.dim() == 1 else a for i, a in enumerate(actions)]

        # --- Stack and restore original batch layout ---
        stacked = torch.stack(actions, dim=1)
        if squeeze_batch:
            stacked = stacked.squeeze(0)
        return stacked

    # ------------------------------------------------------------------
    # RL (TensorDict) group routing
    # ------------------------------------------------------------------

    @staticmethod
    def _run_rl_group(
        input_td: TensorDictBase,
        group: str,
        policy: TensorDictModuleBase,
    ) -> TensorDictBase:
        output_td = policy(input_td)
        if not isinstance(output_td, TensorDictBase):
            raise TypeError(f"RL policy for group {group!r} must return TensorDictBase, got {type(output_td).__name__}.")
        if (group, "action") not in output_td.keys(include_nested=True):
            raise KeyError(f"RL policy for group {group!r} did not write ({group!r}, 'action') to the output TensorDict.")
        return output_td

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_td: TensorDictBase) -> TensorDictBase:
        active_groups = self._resolve_active_groups(input_td)
        output_td = input_td
        for group in active_groups:
            if group in self._rl_groups:
                output_td = self._run_rl_group(output_td, group, self._policies[group])
            else:
                action = self._run_tensor_group(output_td, group, self._policies[group])
                output_td.set((group, "action"), action)
        return output_td
