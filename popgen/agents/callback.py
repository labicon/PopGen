"""BenchMARL callbacks for PopGen policy wiring."""

from __future__ import annotations

import functools
import logging

import torch.nn as nn
from benchmarl.experiment.callback import Callback

from .router import MultiPolicyRouter

logging.basicConfig(level=logging.INFO)


class TeammatePolicyCallback(Callback):
    """BenchMARL callback that routes an RL ego policy with scripted teammates.

    ``on_setup`` fires after ``_setup_collector`` and ``_setup_task``, so
    the experiment has already created its envs, collector, and per-group
    policies.

    The callback wraps ``experiment.group_policies[ego_group]`` with
    :class:`~popgen.agents.router.MultiPolicyRouter` so collection and
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
        wrapped = MultiPolicyRouter(policies={"ego": ego_policy, "teammate": teammate_policies})
        self.experiment.policy = wrapped
        if hasattr(self.experiment, "collector"):
            self.experiment.collector.policy = wrapped
        if hasattr(self.experiment, "test_env"):
            self._patch_rollout_for_trust_policy(self.experiment.test_env)
        if hasattr(self.experiment, "rollout_env"):
            self._patch_rollout_for_trust_policy(self.experiment.rollout_env)

        if self.trainable_group is not None:
            self.experiment.train_group_map = self.trainable_group
