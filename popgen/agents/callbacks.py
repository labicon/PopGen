"""BenchMARL callbacks for training an ego agent against teammate policies."""

from __future__ import annotations

import functools
import logging
import random

import torch
import torch.nn as nn
from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase

from .router import PolicyRouterBase, RLEgoTeammateRouter

logging.basicConfig(level=logging.INFO)


class TeammatePolicyCallback(Callback):
    """BenchMARL callback that routes an RL ego policy with scripted teammates.

    ``on_setup`` fires after ``_setup_collector`` and ``_setup_task``, so
    the experiment has already created its envs, collector, and per-group
    policies.

    The callback wraps ``experiment.group_policies[ego_group]`` with
    :class:`~popgen.agents.router.RLEgoTeammateRouter` so collection and
    evaluation both use:
    - the BenchMARL TensorDict ``ego`` policy unchanged, and
    - tensor-only teammate policies routed from ``teammate`` observations.

    It also re-applies ``trainable_group`` to ``experiment.train_group_map``
    because ``_setup_task`` overwrites that attribute with the full group_map
    before this callback fires.

    BenchMARL evaluation uses ``env.rollout(policy=experiment.policy)``. Since
    our router is a plain ``nn.Module`` rather than a TensorDictModule, the
    callback patches evaluation-style rollout entrypoints to pass
    ``trust_policy=True``.
    """

    def __init__(
        self,
        teammate_policies: list[nn.Module],
        trainable_group: dict | None = None,
    ):
        super().__init__()
        self.teammate_policies = teammate_policies
        self.trainable_group = trainable_group

    @staticmethod
    def _patch_rollout_for_trust_policy(env) -> None:
        rollout = getattr(env, "rollout", None)
        if rollout is None or getattr(rollout, "_popgen_trust_policy_patched", False):
            return

        @functools.wraps(rollout)
        def trusted_rollout(*args, **kwargs):
            kwargs.setdefault("trust_policy", True)
            return rollout(*args, **kwargs)

        trusted_rollout._popgen_trust_policy_patched = True
        env.rollout = trusted_rollout

    def on_setup(self):
        ego_group = next(iter(self.trainable_group.keys())) if self.trainable_group is not None else next(iter(self.experiment.group_map.keys()))
        ego_policy = self.experiment.group_policies[ego_group]

        teammate_policies = self.teammate_policies
        if isinstance(teammate_policies, nn.Module):
            teammate_policies = [teammate_policies]

        logging.info(f"Using teammate policies {teammate_policies} with ego policy {ego_policy} for group '{ego_group}'.")
        wrapped = RLEgoTeammateRouter(
            ego_policy=ego_policy,
            teammate_policies=teammate_policies,
        )
        self.experiment.policy = wrapped
        if hasattr(self.experiment, "collector"):
            self.experiment.collector.policy = wrapped
        if hasattr(self.experiment, "test_env"):
            self._patch_rollout_for_trust_policy(self.experiment.test_env)
        if hasattr(self.experiment, "rollout_env"):
            self._patch_rollout_for_trust_policy(self.experiment.rollout_env)

        if self.trainable_group is not None:
            self.experiment.train_group_map = self.trainable_group


class PopulationCallback(Callback):
    """Train an ego policy against teammate policy populations via semantic routing.

    This callback assumes the same semantic split-group layout as
    :class:`TeammatePolicyCallback`: the trainable RL policy is the ``"ego"``
    group and teammates are routed from ``"teammate"`` observations.

    ``teammate_policies`` is one pool per teammate slot, and this callback
    always assigns teammate policies independently across the parallel
    environments in each batch.
    """

    def __init__(
        self,
        teammate_policies: list[list[nn.Module]],
        trainable_group: dict | None = None,
    ):
        super().__init__()
        self.teammate_policies = teammate_policies
        self.trainable_group = trainable_group
        self._wrapper: PopulationRouter | None = None

    @staticmethod
    def _patch_rollout_for_trust_policy(env) -> None:
        rollout = getattr(env, "rollout", None)
        if rollout is None or getattr(rollout, "_popgen_trust_policy_patched", False):
            return

        @functools.wraps(rollout)
        def trusted_rollout(*args, **kwargs):
            kwargs.setdefault("trust_policy", True)
            return rollout(*args, **kwargs)

        trusted_rollout._popgen_trust_policy_patched = True
        env.rollout = trusted_rollout

    def on_setup(self):
        ego_group = next(iter(self.trainable_group.keys())) if self.trainable_group is not None else next(iter(self.experiment.group_map.keys()))
        ego_policy = self.experiment.group_policies[ego_group]

        self._wrapper = PopulationRouter(
            ego_policy=ego_policy,
            teammate_pools=self.teammate_policies,
        )
        logging.info(f"PopulationCallback: using per-env router-based teammate sampling ({len(self.teammate_policies)} teammate slot(s)) for group '{ego_group}'.")

        self.experiment.policy = self._wrapper
        if hasattr(self.experiment, "collector"):
            self.experiment.collector.policy = self._wrapper
        if hasattr(self.experiment, "test_env"):
            self._patch_rollout_for_trust_policy(self.experiment.test_env)
        if hasattr(self.experiment, "rollout_env"):
            self._patch_rollout_for_trust_policy(self.experiment.rollout_env)

        if self.trainable_group is not None:
            self.experiment.train_group_map = self.trainable_group

    def on_batch_collected(self, _batch):
        if self._wrapper is None:
            return
        self._wrapper.resample()


class PrioritizedCallback(Callback):
    """Train ego against a population using MEP rank-based prioritized sampling.

    Each parallel environment independently samples a teammate policy from the
    pool according to a priority distribution that favours policies with which
    the ego agent achieves *lower* returns (harder partners).  The priority is
    updated **once per batch** using the rewards collected in that batch.

    Sampling probability (MEP, AAAI 2023):

        p(i)  ∝  rank_{1/E[R_i]}^β

    where ``E[R_i]`` is the average per-batch reward obtained when ego plays
    with teammate policy *i*, and rank is assigned so that the policy with the
    *lowest* return gets rank 1 (highest priority).

    Episode-boundary resampling is intentionally disabled; policies are only
    reassigned between batches via :meth:`on_batch_collected`.

    Args:
        teammate_policies: One pool per teammate slot.  ``teammate_policies[i]``
            is the list of candidate ``nn.Module`` policies for teammate *i*.
            For a single teammate use a one-element outer list, e.g.
            ``[[policy_a, policy_b]]``.
        trainable_group: ``{ego_group_name: [agent_names]}`` mapping.
            Re-applied to ``experiment.train_group_map`` in ``on_setup``.
        beta: Priority exponent.  ``beta=0`` gives uniform sampling;
            larger values concentrate more mass on difficult partners.
            Defaults to ``1.0``.
        warmup_batches: Number of batches to sample uniformly before applying
            MEP prioritization. Defaults to 10.
    """

    def __init__(
        self,
        teammate_policies: list[list[nn.Module]],
        trainable_group: dict | None = None,
        beta: float = 1.0,
        warmup_batches: int = 30,
    ):
        super().__init__()
        self.teammate_policies = teammate_policies
        self.trainable_group = trainable_group
        self.beta = beta
        self.warmup_batches = warmup_batches
        self._batches_collected: int = 0
        self._wrapper: PopulationRouter | None = None
        # Return buffer: policy_idx -> list of per-env batch rewards collected since last update.
        self._return_buffer: dict[int, list[float]] = {}
        # Last measured average return per policy (None until first observation).
        self._measured_returns: list[float | None] = []
        # Current sampling probability distribution over policies (per slot, shared across slots).
        self._sampling_probs: list[float] = []

        # Initialise buffers based on the first slot's pool size (all slots share the same prior).
        if teammate_policies:
            n = len(teammate_policies[0])
            self._return_buffer = {i: [] for i in range(n)}
            self._measured_returns = [None] * n
            self._sampling_probs = [1.0 / n] * n

    @staticmethod
    def _patch_rollout_for_trust_policy(env) -> None:
        rollout = getattr(env, "rollout", None)
        if rollout is None or getattr(rollout, "_popgen_trust_policy_patched", False):
            return

        @functools.wraps(rollout)
        def trusted_rollout(*args, **kwargs):
            kwargs.setdefault("trust_policy", True)
            return rollout(*args, **kwargs)

        trusted_rollout._popgen_trust_policy_patched = True
        env.rollout = trusted_rollout

    # ------------------------------------------------------------------
    # Priority computation
    # ------------------------------------------------------------------

    def _compute_sampling_probs(self) -> list[float]:
        """Compute MEP rank-based sampling probabilities.

        Returns uniform distribution during warmup, when any policy has no
        return estimate yet, or when all estimates are identical.
        """
        n = len(self._measured_returns)
        if self._batches_collected <= self.warmup_batches:
            return [1.0 / n] * n

        returns = self._measured_returns

        if any(r is None for r in returns):
            return [1.0 / n] * n

        if len(set(returns)) == 1:
            return [1.0 / n] * n

        # Rank by ascending return: lowest return (hardest partner) gets rank n (highest priority).
        sorted_desc = sorted(range(n), key=lambda i: returns[i], reverse=True)
        ranks = [0] * n
        for rank_val, idx in enumerate(sorted_desc, start=1):
            ranks[idx] = rank_val

        raw = [float(ranks[i] ** self.beta) for i in range(n)]
        total = sum(raw)
        return [p / total for p in raw]

    @staticmethod
    def _extract_per_env_batch_reward(batch) -> torch.Tensor | None:
        """Extract per-environment summed reward from a collected batch TensorDict."""
        nested_keys = set(batch.keys(include_nested=True))
        reward_key = None
        if ("next", "ego", "reward") in nested_keys:
            reward_key = ("next", "ego", "reward")
        elif ("ego", "reward") in nested_keys:
            reward_key = ("ego", "reward")
        if reward_key is None:
            return None

        reward = batch[reward_key]
        while reward.dim() > 1 and reward.shape[-1] == 1:
            reward = reward.squeeze(-1)
        while reward.dim() > 1:
            reward = reward.sum(dim=-1)
        return reward.float()

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_setup(self):
        ego_group = next(iter(self.trainable_group.keys())) if self.trainable_group is not None else next(iter(self.experiment.group_map.keys()))
        ego_policy = self.experiment.group_policies[ego_group]

        self._wrapper = PopulationRouter(
            ego_policy=ego_policy,
            teammate_pools=self.teammate_policies,
            sampling_probs=self._sampling_probs,
            resample_on_done=False,
        )
        logging.info(f"PrioritizedCallback: using prioritized per-env router-based teammate sampling ({len(self.teammate_policies)} teammate slot(s)) for group '{ego_group}'.")

        self.experiment.policy = self._wrapper
        if hasattr(self.experiment, "collector"):
            self.experiment.collector.policy = self._wrapper
        if hasattr(self.experiment, "test_env"):
            self._patch_rollout_for_trust_policy(self.experiment.test_env)
            self._patch_rollout_for_eval_mode(self.experiment.test_env)
        if hasattr(self.experiment, "rollout_env"):
            self._patch_rollout_for_trust_policy(self.experiment.rollout_env)
            self._patch_rollout_for_eval_mode(self.experiment.rollout_env)

        if self.trainable_group is not None:
            self.experiment.train_group_map = self.trainable_group

    def _patch_rollout_for_eval_mode(self, env) -> None:
        """Wrap env.rollout to switch the router to uniform sampling during evaluation."""
        rollout = getattr(env, "rollout", None)
        if rollout is None or getattr(rollout, "_popgen_eval_mode_patched", False):
            return
        wrapper = self._wrapper

        @functools.wraps(rollout)
        def eval_mode_rollout(*args, **kwargs):
            wrapper.eval()
            try:
                return rollout(*args, **kwargs)
            finally:
                wrapper.train()

        eval_mode_rollout._popgen_eval_mode_patched = True
        env.rollout = eval_mode_rollout

    def on_batch_collected(self, batch):
        """Update the priority prior from batch rewards, then resample policies."""
        if self._wrapper is None or self._wrapper._env_policy_indices is None:
            return

        self._batches_collected += 1

        env_policy_indices = self._wrapper._env_policy_indices
        n_envs = len(env_policy_indices[0])

        # --- Step 1: extract per-env reward from the collected batch ---
        per_env_reward = self._extract_per_env_batch_reward(batch)
        if per_env_reward is None:
            logging.warning("PrioritizedPolicyPopulationCallback: no reward key found in batch; skipping prior update and using current probs for resample.")
        else:
            # --- Step 2: accumulate rewards into per-policy return buffers ---
            # Only the first teammate slot is used to attribute rewards; all slots
            # share the same prior distribution.
            slot_indices = env_policy_indices[0]
            rewards_cpu = per_env_reward.cpu().tolist()
            for env_idx, policy_idx in enumerate(slot_indices):
                if env_idx < len(rewards_cpu):
                    self._return_buffer[policy_idx].append(rewards_cpu[env_idx])

            # --- Step 3: update measured returns and reset buffer ---
            n = len(self._measured_returns)
            for i in range(n):
                buf = self._return_buffer.get(i, [])
                if buf:
                    self._measured_returns[i] = sum(buf) / len(buf)

            self._return_buffer = {i: [] for i in range(n)}

            ret_str = ", ".join(f"p{i}={'N/A' if self._measured_returns[i] is None else f'{self._measured_returns[i]:.3f}'}" for i in range(n))
            logging.info(f"PrioritizedPolicyPopulationCallback: measured_returns=[{ret_str}]")

            if hasattr(self.experiment, "logger"):
                log_dict = {f"prioritized/measured_return/policy_{i}": v for i, v in enumerate(self._measured_returns) if v is not None}
                self.experiment.logger.log(log_dict, step=self.experiment.n_iters_performed)

        # --- Step 4: recompute probs, push to router, resample ---
        self._sampling_probs = self._compute_sampling_probs()
        self._wrapper.sampling_probs = self._sampling_probs
        self._wrapper.resample()

        prob_str = ", ".join(f"p{i}={p:.3f}" for i, p in enumerate(self._sampling_probs))
        warmup_str = f" [warmup {self._batches_collected}/{self.warmup_batches}]" if self._batches_collected <= self.warmup_batches else ""
        logging.info(f"PrioritizedPolicyPopulationCallback: resampled {n_envs} envs with probs=[{prob_str}]{warmup_str}")

        if hasattr(self.experiment, "logger"):
            prob_log = {f"prioritized/sampling_prob/policy_{i}": p for i, p in enumerate(self._sampling_probs)}
            self.experiment.logger.log(prob_log, step=self.experiment.n_iters_performed)


class TeamRewardSumCallback(Callback):
    """Adds teammate group reward to ego reward before training.

    BenchMARL trains only the ego group, using ("next", "ego", "reward")
    for the loss. This callback sums the teammate reward into the ego reward
    so the ego policy maximises total team reward instead of solo reward.
    No-op if either reward key is absent from the batch.
    """

    def on_batch_collected(self, batch) -> None:
        nested_keys = set(batch.keys(include_nested=True))
        ego_key = ("next", "ego", "reward")
        tm_key = ("next", "teammate", "reward")
        if ego_key not in nested_keys or tm_key not in nested_keys:
            return

        ego_rew = batch[ego_key]  # (..., n_ego_agents, 1)
        tm_rew = batch[tm_key]  # (..., n_tm_agents, 1)

        # Sum over the agent dimension (second-to-last), keep shape (..., 1, 1)
        # so it broadcasts correctly onto ego_rew of shape (..., n_ego_agents, 1).
        tm_sum = tm_rew.sum(dim=-2, keepdim=True)
        batch[ego_key] = ego_rew + tm_sum

        logging.debug(
            "TeamRewardSumCallback: ego %s + teammate %s",
            tuple(ego_rew.shape),
            tuple(tm_rew.shape),
        )


def select_teammate_callback_cls(*, is_population: bool, prioritized: bool) -> type[Callback]:
    """Choose the teammate-routing callback for fixed or population partners."""
    if not is_population:
        return TeammatePolicyCallback
    if prioritized:
        return PrioritizedCallback
    return PopulationCallback


class PopulationRouter(PolicyRouterBase):
    """Private router that preserves per-env teammate diversity.

    This helper always assigns teammate policies per environment and per
    teammate slot while using semantic ``"ego"`` / ``"teammate"`` routing.
    """

    def __init__(
        self,
        ego_policy: TensorDictModuleBase,
        teammate_pools: list[list[nn.Module]],
        sampling_probs: list[float] | None = None,
        resample_on_done: bool = True,
    ):
        super().__init__()
        if not isinstance(ego_policy, TensorDictModuleBase):
            raise TypeError(f"PopulationCallback expected ego_policy to be TensorDictModuleBase, got {type(ego_policy).__name__}.")
        self.ego_policy = ego_policy
        self.teammate_pools = teammate_pools
        self.sampling_probs = sampling_probs  # None = uniform; list[float] for weighted
        # When False, teammate assignments are held fixed within a collection batch
        # (episode-boundary resampling disabled). MEP prioritized sampling requires
        # this so per-batch reward attribution matches the teammate that produced it.
        self.resample_on_done = resample_on_done
        self._policy_groups: list[list[tuple[nn.Module, list[int], torch.Tensor]]] | None = None  # [slot] → [(policy, env_indices, idx_tensor)]
        self._env_policy_indices: list[list[int]] | None = None  # [slot][env] policy indices

    def _sample_slot_indices(self, n_envs: int) -> list[int]:
        """Sample n_envs policy indices for one slot using sampling_probs (or uniform).

        Uses uniform sampling when in eval mode (self.training is False) so that
        evaluation metrics are not biased toward the hardest-partner distribution.
        """
        n_pool = len(self.teammate_pools[0])
        probs = self.sampling_probs if self.training else None
        if probs is not None:
            probs_t = torch.tensor(probs, dtype=torch.float32)
            return torch.multinomial(probs_t, num_samples=n_envs, replacement=True).tolist()
        return [random.randrange(n_pool) for _ in range(n_envs)]

    def _build_policy_groups(self, assignments: list[list[int]]) -> list[list[tuple[nn.Module, list[int], torch.Tensor]]]:
        """Group env indices by assigned policy index for each slot.

        Each entry is a (policy, env_indices, idx_tensor) triple. The idx_tensor
        is a pre-built CPU LongTensor so _run_teammate_group avoids recreating it
        on every step (it only changes when resample is called).
        """
        from collections import defaultdict

        policy_groups = []
        for pool, slot_indices in zip(self.teammate_pools, assignments):
            groups: dict[int, list[int]] = defaultdict(list)
            for env_idx, pol_idx in enumerate(slot_indices):
                groups[pol_idx].append(env_idx)
            policy_groups.append([(pool[pol_idx], env_idxs, torch.tensor(env_idxs, dtype=torch.long)) for pol_idx, env_idxs in groups.items()])
        return policy_groups

    def _init_env_policies(self, n_envs: int) -> None:
        assignments = [self._sample_slot_indices(n_envs) for _ in self.teammate_pools]
        self._policy_groups = self._build_policy_groups(assignments)
        self._env_policy_indices = assignments
        logging.info(
            "PopulationRouter: initialised %d env x %d teammate-slot assignments.",
            n_envs,
            len(self.teammate_pools),
        )

    def train(self, mode: bool = True) -> PopulationRouter:
        """Override to resample with the appropriate distribution on mode change.

        Eval mode (mode=False) resamples uniformly; train mode resamples using
        sampling_probs so evaluation metrics reflect all policies equally.
        """
        was_training = self.training
        super().train(mode)
        if self._policy_groups is not None and was_training != mode:
            self.resample()
        return self

    def resample(self) -> None:
        if self._policy_groups is None:
            return
        n_envs = sum(len(env_idxs) for _, env_idxs, _ in self._policy_groups[0])
        assignments = [self._sample_slot_indices(n_envs) for _ in self.teammate_pools]
        self._policy_groups = self._build_policy_groups(assignments)
        self._env_policy_indices = assignments

    def _resample_done_envs(self, done_flat: torch.Tensor) -> None:
        """Resample policies for environments whose episode just ended.

        For each done environment, draws a fresh policy index (respecting
        ``sampling_probs``) and updates both ``_env_policy_indices`` and
        ``_policy_groups``.  Environments that are not done keep their
        current assignment unchanged.

        ``_policy_groups`` is rebuilt from scratch after updating the index
        list, which keeps ``_build_policy_groups`` as the single source of
        truth for the grouping structure.
        """
        if self._policy_groups is None or self._env_policy_indices is None:
            return
        done_list = done_flat.view(-1).bool().cpu().tolist()
        if not any(done_list):
            return

        probs_t = torch.tensor(self.sampling_probs, dtype=torch.float32) if self.training and self.sampling_probs is not None else None
        n_pool = len(self.teammate_pools[0])

        for slot_idx in range(len(self.teammate_pools)):
            for env_idx, done in enumerate(done_list):
                if done:
                    new_idx = int(torch.multinomial(probs_t, num_samples=1).item()) if probs_t is not None else random.randrange(n_pool)
                    self._env_policy_indices[slot_idx][env_idx] = new_idx

        self._policy_groups = self._build_policy_groups(self._env_policy_indices)

    def _run_teammate_group(
        self,
        *,
        input_td,
        group: str,
    ) -> torch.Tensor:
        """Run all teammate policies for one group and return stacked actions.

        Instead of calling each policy once per environment (O(n_envs) calls per
        slot), this method groups environments by their assigned policy and calls
        each distinct policy once on its combined batch of observations:

            obs[env_indices, slot, :]  →  policy  →  action[env_indices]

        Results are scattered back into a full ``(batch_size, action_dim)`` tensor
        before being stacked across slots.  For a pool of *k* policies and *N* envs,
        this reduces per-slot policy calls from *N* to at most *k* (typically far
        fewer when envs are spread evenly).
        """
        obs, squeeze_batch_dim = self._get_group_observation(input_td, group)
        batch_size, n_agents, _ = obs.shape
        obs_device = obs.device

        if len(self.teammate_pools) != n_agents:
            raise ValueError(f"Expected {n_agents} teammate policy pools for semantic 'teammate' routing, got {len(self.teammate_pools)}.")

        if self._policy_groups is None or sum(len(e) for _, e, _ in self._policy_groups[0]) != batch_size:
            self._init_env_policies(batch_size)

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

        for slot_idx in range(n_agents):
            group_results: list[tuple[list[int], torch.Tensor]] = []

            for policy, env_indices, idx_t in self._policy_groups[slot_idx]:
                idx_t = idx_t.to(obs_device)  # no-op when already on device
                batch_obs = obs[idx_t, slot_idx, :].to(self._get_policy_device(policy))
                action = policy(batch_obs).to(obs_device)
                action, action_rank, action_dim = self._validate_action_tensor(
                    action,
                    batch_size=len(env_indices),
                    policy_index=slot_idx,
                    expected_rank=action_rank,
                    expected_action_dim=action_dim,
                )
                if action_mask is not None and action.dim() == 1:
                    action = self._apply_action_mask(action, action_mask[idx_t, slot_idx, :])
                group_results.append((idx_t, action))

            # Scatter results back into a full-batch slot tensor
            first = group_results[0][1]
            if action_rank == 1:
                slot_action = torch.empty(batch_size, dtype=first.dtype, device=obs_device)
            else:
                slot_action = torch.empty(batch_size, action_dim, dtype=first.dtype, device=obs_device)
            for idx_t, action in group_results:
                slot_action[idx_t] = action
            action_list.append(slot_action)

        return self._stack_group_actions(action_list, squeeze_batch_dim=squeeze_batch_dim)

    def forward(self, input_td):
        controlled_groups = self._resolve_controlled_groups(input_td)
        if "ego" not in controlled_groups:
            raise ValueError("PopulationRouter requires semantic 'ego' and optional 'teammate' groups.")

        output_td = self.ego_policy(input_td)
        if not isinstance(output_td, TensorDictBase):
            raise TypeError(f"RL ego policy must return TensorDictBase, got {type(output_td).__name__}.")
        if ("ego", "action") not in output_td.keys(include_nested=True):
            raise KeyError("RL ego policy did not write ('ego', 'action') to the output TensorDict.")

        if "teammate" in controlled_groups:
            teammate_obs, _ = self._get_group_observation(output_td, "teammate")
            n_envs = teammate_obs.shape[0]
            if self._policy_groups is None or sum(len(e) for _, e, _ in self._policy_groups[0]) != n_envs:
                self._init_env_policies(n_envs)

            if self.resample_on_done:
                for done_key in ("done", "terminated"):
                    done = output_td.get(done_key, input_td.get(done_key, None))
                    if done is not None:
                        self._resample_done_envs(done.view(n_envs))
                        break

            teammate_action = self._run_teammate_group(input_td=output_td, group="teammate")
            output_td.set(("teammate", "action"), teammate_action)
        elif len(self.teammate_pools) > 0:
            raise ValueError(f"Received {len(self.teammate_pools)} teammate policy pools but rollout TensorDict has no 'teammate' group.")

        return output_td
