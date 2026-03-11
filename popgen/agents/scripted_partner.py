"""Scripted partner wrapper utilities for BenchMARL policies."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from benchmarl.experiment.callback import Callback
from tensordict import TensorDict

from .utils import find_observation_group

logging.basicConfig(level=logging.INFO)


def _normalize_stacked_action(action) -> torch.Tensor:
    """Normalize action shape before stacking per-agent outputs."""
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action)

    if action.dim() == 0:
        return action

    # Preserve per-policy semantics while removing singleton agent axis
    # introduced by grouped slicing (..., i:i+1, ...).
    if action.dim() >= 2 and action.shape[-2] == 1:
        action = action.squeeze(-2)

    # Grouped single-agent slicing in discrete environments can leave a
    # singleton 1-D tensor like [4]; collapse it back to a scalar before
    # stacking over agents.
    if action.dim() == 1 and action.shape[0] == 1:
        action = action.squeeze(0)

    # Keep 1D tensors (e.g. scalar-discrete batch) intact; only strip a
    # trailing singleton feature dim when one truly exists.
    if action.dim() > 1 and action.shape[-1] == 1:
        action = action.squeeze(-1)

    return action


def _extract_action(policy_output, group: str | None = None) -> torch.Tensor:
    """Extract the action tensor from a policy's output.

    Accepts either a raw ``torch.Tensor`` (returned directly) or a
    TensorDict-like object from which the key ``(group, "action")``
    is read.

    Args:
        policy_output: Output produced by calling a policy module.
        group: Group name used to look up the action key when
            *policy_output* is a TensorDict (e.g. ``"ego"``, ``"agents"``).

    Returns:
        The action tensor for *group*.

    Raises:
        TypeError: If *policy_output* is neither a tensor nor TensorDict-like.
        KeyError: If *policy_output* is TensorDict-like but does not contain
            the expected ``(group, "action")`` key.
    """
    if isinstance(policy_output, torch.Tensor):
        return policy_output
    if not hasattr(policy_output, "keys"):
        raise TypeError("Policy output must be a torch.Tensor or TensorDict-like.")
    nested_keys = list(policy_output.keys(include_nested=True))
    if group is not None:
        key = (group, "action")
        if key in nested_keys:
            return policy_output[key]

    if "action" in policy_output.keys():
        return policy_output["action"]

    action_keys = [key for key in nested_keys if isinstance(key, tuple) and len(key) == 2 and key[1] == "action"]
    if len(action_keys) == 1:
        return policy_output[action_keys[0]]

    expected = f"{(group, 'action')!r}" if group is not None else "'action'"
    raise KeyError(f"Policy output missing expected action key {expected}. Available: {nested_keys}")


class MultiPolicyWrapper(nn.Module):
    """Route a shared TensorDict through one policy per agent.

    ``policies[i]`` is applied to agent *i* across the whole team.  Two
    observation-group layouts are supported:

    1. **Split ego/teammate groups** – ``("ego", "observation")`` and
       ``("teammate", "observation")`` are separate top-level keys, each
       shaped ``[batch, n_agents_in_group, obs_dim]``.  ``policies[0]``
       handles the single ego agent; ``policies[1], policies[2], ...``
       handle teammates in order.
    2. **Single shared group** (e.g. ``"agents"``) – all agents are stacked
       along the ``-2`` axis; ``policies[i]`` handles agent *i*.

    In both layouts the wrapper slices one agent at a time out of the group
    TensorDict, runs its policy, then assembles the full action buffer.
    """

    def __init__(self, policies: list[nn.Module]):
        super().__init__()
        self.policies = policies

    def _run_group(self, input_td: TensorDict, group: str, policy_offset: int) -> torch.Tensor:
        """Slice each agent in *group*, run its policy, return the assembled action tensor.

        Args:
            input_td: The full-batch TensorDict (mutated temporarily per agent, then restored).
            group: Group key (e.g. ``"ego"`` or ``"agents"``).
            policy_offset: Index into ``self.policies`` for the first agent in this group.

        Returns:
            Action tensor shaped ``[*batch, n_agents_in_group, action_dim]`` or
            ``[*batch, n_agents_in_group]`` for scalar actions.
        """
        full_group_td = input_td[group]
        obs = full_group_td["observation"]
        n = obs.shape[-2]

        if policy_offset + n > len(self.policies):
            raise ValueError(f"Group '{group}' has {n} agents starting at policy index {policy_offset}, but only {len(self.policies)} policies were provided.")

        grouped_keys = [key for key in input_td.keys() if isinstance(key, str) and (key, "observation") in input_td.keys(include_nested=True)]

        action_list = []
        for i in range(n):
            # Preserve non-group rollout keys (e.g. is_init/reset masks) needed
            # by recurrent policies while slicing a single agent in the target group.
            per_agent_payload = {}
            for key in input_td.keys():
                if key == group:
                    per_agent_payload[group] = full_group_td[..., i : i + 1]
                elif key in grouped_keys:
                    continue
                else:
                    per_agent_payload[key] = input_td[key]

            per_agent_td = TensorDict(
                per_agent_payload,
                batch_size=input_td.batch_size,
                device=input_td.device,
            )
            action = _extract_action(self.policies[policy_offset + i](per_agent_td), group=group)
            action_list.append(_normalize_stacked_action(action))

        return torch.stack(action_list, dim=obs.dim() - 2)

    def _run_flat_observation(self, input_td: TensorDict) -> torch.Tensor:
        """Run per-agent policies for flat TensorDict layout with top-level observation.

        Expected input layout:
        - ``input_td["observation"]`` has shape ``[*batch, n_agents, obs_dim]``
        - agent axis is the ``-2`` dimension.
        """
        obs = input_td["observation"]
        n_agents = obs.shape[-2]

        if len(self.policies) != n_agents:
            raise ValueError(f"Expected {n_agents} policies for flat observation layout, got {len(self.policies)}.")

        action_list = []
        per_agent_batch = input_td.batch_size[:-1]
        for i in range(n_agents):
            per_agent_payload = {"observation": obs[..., i, :]}
            for key in input_td.keys():
                if key == "observation":
                    continue
                per_agent_payload[key] = input_td[key]
            per_agent_td = TensorDict(
                per_agent_payload,
                batch_size=per_agent_batch,
                device=input_td.device,
            )
            action = _extract_action(self.policies[i](per_agent_td), group=None)
            action_list.append(_normalize_stacked_action(action))

        return torch.stack(action_list, dim=obs.dim() - 2)

    def forward(self, input_td: TensorDict):
        """Run all per-agent policies and write actions into *input_td*.

        Args:
            input_td: Batch TensorDict for the current environment step.

        Returns:
            The same *input_td* TensorDict with action keys populated for each
            agent group.

        Raises:
            ValueError: If ``len(self.policies)`` does not match the total
                number of agents found in *input_td*.
        """
        nested_keys = input_td.keys(include_nested=True)

        grouped_observation_keys = [key for key in input_td.keys() if isinstance(key, str) and (key, "observation") in nested_keys]
        if grouped_observation_keys:
            # Preserve deterministic policy assignment order and keep ego first
            # when present so policy[0] is always the ego policy.
            if "ego" in grouped_observation_keys:
                ordered_groups = ["ego"] + [k for k in grouped_observation_keys if k != "ego"]
            else:
                ordered_groups = grouped_observation_keys

            group_counts = [input_td[g]["observation"].shape[-2] for g in ordered_groups]
            total_agents = int(sum(group_counts))
            if len(self.policies) != total_agents:
                counts_str = ", ".join(f"{g}:{c}" for g, c in zip(ordered_groups, group_counts))
                raise ValueError(f"Expected {total_agents} policies for grouped layout ({counts_str}), got {len(self.policies)}.")

            policy_offset = 0
            for group, n_group_agents in zip(ordered_groups, group_counts):
                input_td.set((group, "action"), self._run_group(input_td, group, policy_offset=policy_offset))
                policy_offset += n_group_agents
            return input_td

        # Flat layout used by some VMAS wrappers:
        # TensorDict({"observation": [*batch, n_agents, obs_dim]}, batch_size=[*batch, n_agents]).
        if "observation" in input_td.keys():
            input_td.set("action", self._run_flat_observation(input_td))
            return input_td

        # Single shared group: policy index == agent index.
        group = find_observation_group(input_td)
        n_agents = input_td[group]["observation"].shape[-2]

        if len(self.policies) != n_agents:
            raise ValueError(f"Expected {n_agents} policies for group '{group}', got {len(self.policies)}.")

        input_td.set((group, "action"), self._run_group(input_td, group, policy_offset=0))
        return input_td


class TeammatePolicyCallback(Callback):
    """BenchMARL callback that wraps the trainable policy with teammate policies.

    ``on_setup`` fires after ``_setup_collector`` and ``_setup_task``, so
    ``experiment.collector`` and ``experiment.group_policies`` are available.

    The callback uses ``experiment.group_policies[ego_group]`` (the per-group
    slice of the collection policy) rather than ``experiment.policy`` directly.
    ``experiment.policy`` processes *all* groups at once; passing it a partial
    TensorDict (ego keys only, from ``MultiPolicyWrapper._run_group``) makes it
    look up the missing teammate observation, receive ``None``, and crash with
    ``torch.flatten(None)``.

    It also re-applies ``trainable_group`` to ``experiment.train_group_map``
    because ``_setup_task`` overwrites that attribute with the full group_map
    before this callback fires.
    """

    def __init__(
        self,
        teammate_policies: list[nn.Module],
        trainable_group: dict | None = None,
    ):
        """
        Args:
            teammate_policies: Resolved ``nn.Module`` teammate policies, in
                agent-index order (index 1, 2, …).
            trainable_group: ``{ego_group_name: [agent_names]}`` mapping.
                Re-applied to ``experiment.train_group_map`` in ``on_setup``.
        """
        super().__init__()
        self.teammate_policies = teammate_policies
        self.trainable_group = trainable_group

    def on_setup(self):
        ego_group = next(iter(self.trainable_group.keys())) if self.trainable_group is not None else next(iter(self.experiment.group_map.keys()))
        ego_policy = self.experiment.group_policies[ego_group]

        teammate_policies = self.teammate_policies
        if isinstance(teammate_policies, nn.Module):
            teammate_policies = [teammate_policies]

        logging.info(f"Using teammate policies {teammate_policies} with ego policy {ego_policy} for group '{ego_group}'.")
        wrapped = MultiPolicyWrapper(policies=[ego_policy, *teammate_policies])
        if hasattr(self.experiment, "collector"):
            self.experiment.collector.policy = wrapped

        if self.trainable_group is not None:
            self.experiment.train_group_map = self.trainable_group
