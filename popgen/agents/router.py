"""Policy routers for rollout-time multi-agent routing."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase


class PolicyRouterBase(nn.Module):
    """Shared rollout routing helpers for semantic ``ego`` / ``teammate`` groups."""

    @staticmethod
    def _get_policy_device(policy: nn.Module) -> torch.device:
        """Return the device of *policy*'s first parameter, falling back to CPU."""
        try:
            return next(policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _resolve_controlled_groups(input_td: TensorDictBase) -> list[str]:
        nested_keys = set(input_td.keys(include_nested=True))
        observation_groups = [key for key in input_td.keys() if isinstance(key, str) and (key, "observation") in nested_keys]
        controlled_groups = [group for group in ("ego", "teammate") if group in observation_groups]
        unsupported_groups = [group for group in observation_groups if group not in {"ego", "teammate", "opponents"}]

        if unsupported_groups or not controlled_groups:
            raise ValueError(f"Policy routers only support rollout TensorDicts grouped by 'ego' and optional 'teammate'. Received observation groups {observation_groups}.")

        return controlled_groups

    def _get_group_observation(
        self,
        input_td: TensorDictBase,
        group: str,
    ) -> tuple[torch.Tensor, bool]:
        if group not in input_td.keys():
            raise KeyError(f"Expected group '{group}' in input TensorDict keys {list(input_td.keys())}.")
        group_td = input_td[group]
        if "observation" not in group_td.keys():
            raise KeyError(f"Expected key ({group!r}, 'observation') in input TensorDict.")
        obs = group_td["observation"]
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"Expected ({group!r}, 'observation') to be a torch.Tensor, got {type(obs).__name__}.")
        if obs.dim() == 2:
            return obs.unsqueeze(0), True
        if obs.dim() == 3:
            return obs, False
        raise ValueError(f"Expected ({group!r}, 'observation') with shape (n_agents, obs_dim) or (batch, n_agents, obs_dim), got {tuple(obs.shape)}.")

    @staticmethod
    def _stack_group_actions(
        action_list: list[torch.Tensor],
        *,
        squeeze_batch_dim: bool,
    ) -> torch.Tensor:
        action = torch.stack(action_list, dim=1)
        if squeeze_batch_dim:
            action = action.squeeze(0)
        return action

    @staticmethod
    def _validate_action_tensor(
        action: torch.Tensor,
        *,
        batch_size: int,
        policy_index: int,
        expected_rank: int | None,
        expected_action_dim: int | None,
    ) -> tuple[torch.Tensor, int, int | None]:
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"Policy index {policy_index} must return a torch.Tensor action, got {type(action).__name__}.")
        if action.dim() not in (1, 2):
            raise ValueError(f"Policy index {policy_index} must return action shape (batch,) or (batch, action_dim), got {tuple(action.shape)}.")
        if action.shape[0] != batch_size:
            raise ValueError(f"Policy index {policy_index} returned batch size {action.shape[0]}, expected {batch_size}.")
        if expected_rank is not None and action.dim() != expected_rank:
            raise ValueError(f"Mixed action ranks are not supported: policy index {policy_index} returned rank {action.dim()}, expected {expected_rank}.")

        action_dim = action.shape[1] if action.dim() == 2 else None
        if action.dim() == 2 and expected_action_dim is not None and action_dim != expected_action_dim:
            raise ValueError(f"Inconsistent continuous action dimensions: policy index {policy_index} returned {action_dim}, expected {expected_action_dim}.")
        return action, action.dim(), action_dim

    @staticmethod
    def _apply_action_mask(action: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        """Replace invalid discrete actions with the first valid action per batch slot."""
        action_idx = action.long()
        selected_valid = agent_mask.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        first_valid = agent_mask.float().argmax(dim=1).to(action.dtype)
        return torch.where(selected_valid, action, first_valid)


class EgoTeammateRouter(PolicyRouterBase):
    """Rollout-only router for semantic ``ego`` / ``teammate`` TensorDicts.

    This router only supports input TensorDicts whose observation groups are
    semantic ``"ego"`` and optional ``"teammate"``. ``"opponents"`` may be
    present in the TensorDict but is ignored. Any other observation grouping
    such as ``"agents"`` or ``"agent_blue"`` is rejected with an error.

    Each routed group's ``input_td[group]["observation"]`` may have shape
    ``[batch, n_agents, obs_dim]`` or ``[n_agents, obs_dim]`` for unbatched
    single-env rollouts. Wrapped policies always receive one agent slice at a
    time with shape ``[batch, obs_dim]`` and must return either ``[batch]``
    for discrete actions or ``[batch, action_dim]`` for continuous actions.
    The router stacks those outputs back into ``(group, "action")`` for each
    routed group, preserving the original batched vs unbatched group layout.
    """

    def __init__(self, policies: list[nn.Module]):
        super().__init__()
        self.policies = nn.ModuleList(list(policies))

    def _validate_policy_count(self, *, expected_count: int) -> None:
        if expected_count > len(self.policies):
            raise ValueError(f"Expected at least {expected_count} policies for semantic 'ego'/'teammate' routing, got {len(self.policies)}.")

    def _run_group(
        self,
        *,
        input_td: TensorDictBase,
        group: str,
        policy_offset: int,
    ) -> tuple[torch.Tensor, int]:
        obs, squeeze_batch_dim = self._get_group_observation(input_td, group)
        batch_size, n_agents, _ = obs.shape
        self._validate_policy_count(expected_count=policy_offset + n_agents)
        obs_device = obs.device

        # Extract action mask if present; normalise to (batch, n_agents, n_actions).
        nested_keys = set(input_td.keys(include_nested=True))
        action_mask = None
        if (group, "action_mask") in nested_keys:
            raw_mask = input_td[group, "action_mask"]
            if raw_mask.dim() == 2:
                action_mask = raw_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                action_mask = raw_mask

        action_list = []
        action_rank = None
        action_dim = None

        for i in range(n_agents):
            policy_index = policy_offset + i
            policy = self.policies[policy_index]
            agent_obs = obs[:, i, :].to(self._get_policy_device(policy))
            action = policy(agent_obs)
            action, action_rank, action_dim = self._validate_action_tensor(
                action.to(obs_device),
                batch_size=batch_size,
                policy_index=policy_index,
                expected_rank=action_rank,
                expected_action_dim=action_dim,
            )
            if action_mask is not None and action.dim() == 1:
                action = self._apply_action_mask(action, action_mask[:, i, :].to(obs_device))
            action_list.append(action)

        return self._stack_group_actions(action_list, squeeze_batch_dim=squeeze_batch_dim), n_agents

    def forward(self, input_td: TensorDictBase):
        """Route grouped observations to tensor-only per-agent policies."""
        controlled_groups = self._resolve_controlled_groups(input_td)
        policy_offset = 0
        for group in controlled_groups:
            group_action, n_agents = self._run_group(
                input_td=input_td,
                group=group,
                policy_offset=policy_offset,
            )
            input_td.set((group, "action"), group_action)
            policy_offset += n_agents

        if policy_offset != len(self.policies):
            raise ValueError(f"Expected {policy_offset} policies for semantic 'ego'/'teammate' routing, got {len(self.policies)}.")

        return input_td


class RLEgoTeammateRouter(PolicyRouterBase):
    """Router for RL ``ego`` policy plus tensor-only teammate policies.

    This router only supports rollout TensorDicts grouped as semantic
    ``"ego"`` and optional ``"teammate"``. The RL ``ego_policy`` is called
    with the full incoming TensorDict so group-local extras such as
    ``action_mask`` or recurrent state remain available. Teammate policies are
    assumed to be tensor-only and are routed from teammate observations just
    like :class:`EgoTeammateRouter`.
    """

    def __init__(
        self,
        ego_policy: TensorDictModuleBase,
        teammate_policies: list[nn.Module] | None = None,
    ):
        nn.Module.__init__(self)
        if not isinstance(ego_policy, TensorDictModuleBase):
            raise TypeError(f"RLEgoTeammateRouter expected ego_policy to be TensorDictModuleBase, got {type(ego_policy).__name__}.")
        self.ego_policy = ego_policy
        self.teammate_policies = list(teammate_policies or [])

    def _validate_teammate_policy_count(self, *, expected_count: int) -> None:
        if expected_count != len(self.teammate_policies):
            raise ValueError(f"Expected exactly {expected_count} teammate policies for semantic 'teammate' routing, got {len(self.teammate_policies)}.")

    def _run_teammate_group(
        self,
        *,
        input_td: TensorDictBase,
        group: str,
    ) -> torch.Tensor:
        obs, squeeze_batch_dim = self._get_group_observation(input_td, group)
        batch_size, n_agents, _ = obs.shape
        self._validate_teammate_policy_count(expected_count=n_agents)
        obs_device = obs.device

        nested_keys = set(input_td.keys(include_nested=True))
        action_mask = None
        if (group, "action_mask") in nested_keys:
            raw_mask = input_td[group, "action_mask"]
            if raw_mask.dim() == 2:
                action_mask = raw_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                action_mask = raw_mask

        action_list = []
        action_rank = None
        action_dim = None

        for i, policy in enumerate(self.teammate_policies):
            agent_obs = obs[:, i, :].to(self._get_policy_device(policy))
            action = policy(agent_obs)
            action, action_rank, action_dim = self._validate_action_tensor(
                action.to(obs_device),
                batch_size=batch_size,
                policy_index=i,
                expected_rank=action_rank,
                expected_action_dim=action_dim,
            )
            if action_mask is not None and action.dim() == 1:
                action = self._apply_action_mask(action, action_mask[:, i, :].to(obs_device))
            action_list.append(action)

        return self._stack_group_actions(action_list, squeeze_batch_dim=squeeze_batch_dim)

    def forward(self, input_td: TensorDictBase):
        controlled_groups = self._resolve_controlled_groups(input_td)
        if "ego" not in controlled_groups:
            raise ValueError("RLEgoTeammateRouter requires rollout TensorDicts grouped by semantic 'ego' and optional 'teammate'.")

        output_td = self.ego_policy(input_td)
        if not isinstance(output_td, TensorDictBase):
            raise TypeError(f"RL ego policy must return TensorDictBase, got {type(output_td).__name__}.")
        if ("ego", "action") not in output_td.keys(include_nested=True):
            raise KeyError("RL ego policy did not write ('ego', 'action') to the output TensorDict.")

        has_teammate_group = "teammate" in controlled_groups
        if has_teammate_group:
            teammate_action = self._run_teammate_group(input_td=output_td, group="teammate")
            output_td.set(("teammate", "action"), teammate_action)
        elif len(self.teammate_policies) > 0:
            raise ValueError(f"Received {len(self.teammate_policies)} teammate policies but rollout TensorDict has no 'teammate' group.")

        return output_td


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

    def _resolve_active_groups(self, input_td: TensorDictBase) -> list[str]:
        """Return groups present in both the TensorDict and ``self._policies``."""
        nested_keys = set(input_td.keys(include_nested=True))
        observation_groups = [key for key in input_td.keys() if isinstance(key, str) and (key, "observation") in nested_keys]
        return [g for g in observation_groups if g in self._policies]

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
