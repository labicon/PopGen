"""Unit tests for popgen.agents.router."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from popgen.agents.router import MultiPolicyRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_td(
    groups: list[str],
    *,
    batch: int = 2,
    n_agents: int = 1,
    obs_dim: int = 4,
    with_mask: bool = False,
    n_actions: int = 5,
    batched: bool = True,
) -> TensorDict:
    """Build a minimal grouped TensorDict for router tests."""
    group_data = {}
    for group in groups:
        obs = torch.zeros(batch, n_agents, obs_dim) if batched else torch.zeros(n_agents, obs_dim)
        entry: dict = {"observation": obs}
        if with_mask:
            mask = torch.ones(batch, n_agents, n_actions, dtype=torch.bool) if batched else torch.ones(n_agents, n_actions, dtype=torch.bool)
            entry["action_mask"] = mask
        group_data[group] = entry
    return TensorDict(group_data, batch_size=[batch] if batched else [])


def _const_policy(action_value: int = 0, *, action_dim: int | None = None) -> nn.Module:
    """Return an nn.Module that outputs a constant action tensor for any batch input."""

    class _ConstPolicy(nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            batch = obs.shape[0]
            if action_dim is None:
                return torch.full((batch,), action_value, dtype=torch.long)
            return torch.full((batch, action_dim), float(action_value))

    return _ConstPolicy()


class _MockEgoTDPolicy(TensorDictModuleBase):
    """Minimal TensorDictModuleBase that writes (ego, action) from ego observations."""

    in_keys = [("ego", "observation")]
    out_keys = [("ego", "action")]

    def __init__(self, n_ego_agents: int = 1, *, write_action: bool = True):
        super().__init__()
        self.n_ego_agents = n_ego_agents
        self.write_action = write_action

    def forward(self, input_td: TensorDict) -> TensorDict:
        obs = input_td["ego", "observation"]
        if obs.dim() == 3:
            batch = obs.shape[0]
            action = torch.zeros(batch, self.n_ego_agents, dtype=torch.long)
        else:
            action = torch.zeros(self.n_ego_agents, dtype=torch.long)
        if self.write_action:
            input_td.set(("ego", "action"), action)
        return input_td


# ---------------------------------------------------------------------------
# TestMultiPolicyRouter — tensor (per-agent) groups
# ---------------------------------------------------------------------------


class TestMultiPolicyRouterTensor:
    def test_ego_only_discrete(self):
        td = _make_td(["ego"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(action_value=2)]})
        out = router(td)
        action = out["ego", "action"]
        assert action.shape == (2, 1)
        assert (action == 2).all()

    def test_ego_and_teammate_discrete(self):
        td = _make_td(["ego", "teammate"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0)], "teammate": [_const_policy(1)]})
        out = router(td)
        nested = set(out.keys(include_nested=True))
        assert ("ego", "action") in nested
        assert ("teammate", "action") in nested
        assert out["ego", "action"].shape == (2, 1)
        assert out["teammate", "action"].shape == (2, 1)
        assert (out["ego", "action"] == 0).all()
        assert (out["teammate", "action"] == 1).all()

    def test_observations_routed_to_correct_policy(self):
        """Each policy must receive the observation slice from its own group."""
        batch, obs_dim = 2, 4

        class _RecordingPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.received: list[torch.Tensor] = []

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                self.received.append(obs.clone())
                return torch.zeros(obs.shape[0], dtype=torch.long)

        ego_obs = torch.ones(batch, 1, obs_dim) * 1.0
        teammate_obs = torch.ones(batch, 1, obs_dim) * 2.0
        td = TensorDict(
            {"ego": {"observation": ego_obs}, "teammate": {"observation": teammate_obs}},
            batch_size=[batch],
        )

        ego_policy = _RecordingPolicy()
        teammate_policy = _RecordingPolicy()
        router = MultiPolicyRouter(policies={"ego": [ego_policy], "teammate": [teammate_policy]})
        router(td)

        assert len(ego_policy.received) == 1
        assert torch.allclose(ego_policy.received[0], ego_obs[:, 0, :])

        assert len(teammate_policy.received) == 1
        assert torch.allclose(teammate_policy.received[0], teammate_obs[:, 0, :])

    def test_unbatched_observation(self):
        td = _make_td(["ego"], n_agents=1, batched=False)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0)]})
        out = router(td)
        assert out["ego", "action"].shape == (1,)

    def test_continuous_actions(self):
        td = _make_td(["ego"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0, action_dim=3)]})
        out = router(td)
        assert out["ego", "action"].shape == (2, 1, 3)

    def test_action_mask_corrects_invalid(self):
        batch, n_agents, obs_dim, n_actions = 2, 1, 4, 5
        obs = torch.zeros(batch, n_agents, obs_dim)
        mask = torch.zeros(batch, n_agents, n_actions, dtype=torch.bool)
        mask[:, :, 1] = True
        td = TensorDict({"ego": {"observation": obs, "action_mask": mask}}, batch_size=[batch])
        router = MultiPolicyRouter(policies={"ego": [_const_policy(action_value=3)]})
        out = router(td)
        assert (out["ego", "action"] == 1).all()

    def test_action_mask_2d_promoted(self):
        """2-D mask (n_agents, n_actions) in an unbatched TensorDict is promoted to 3-D."""
        n_agents, obs_dim, n_actions = 1, 4, 5
        obs = torch.zeros(n_agents, obs_dim)
        mask = torch.ones(n_agents, n_actions, dtype=torch.bool)
        td = TensorDict({"ego": {"observation": obs, "action_mask": mask}}, batch_size=[])
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0)]})
        out = router(td)
        assert out["ego", "action"].shape == (n_agents,)

    def test_too_few_policies_raises(self):
        td = _make_td(["ego"], batch=2, n_agents=2)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0)]})
        with pytest.raises(ValueError, match="2 agents but 1 policies"):
            router(td)

    def test_mixed_action_ranks_raises(self):
        td = _make_td(["ego"], batch=2, n_agents=2)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0), _const_policy(0, action_dim=3)]})
        with pytest.raises(ValueError, match="Mixed action ranks"):
            router(td)

    def test_inconsistent_action_dim_raises(self):
        td = _make_td(["ego"], batch=2, n_agents=2)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0, action_dim=3), _const_policy(0, action_dim=4)]})
        with pytest.raises(ValueError, match="Inconsistent continuous action dimensions"):
            router(td)

    def test_multi_agent_group(self):
        """Multiple agents in a single tensor group, each with distinct policy."""
        td = _make_td(["ego"], batch=2, n_agents=3)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(10), _const_policy(20), _const_policy(30)]})
        out = router(td)
        assert out["ego", "action"].shape == (2, 3)
        assert (out["ego", "action"][:, 0] == 10).all()
        assert (out["ego", "action"][:, 1] == 20).all()
        assert (out["ego", "action"][:, 2] == 30).all()

    def test_too_many_policies_raises(self):
        td = _make_td(["ego"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0), _const_policy(0)]})
        with pytest.raises(ValueError, match="1 agents but 2 policies"):
            router(td)

    def test_action_mask_multi_agent(self):
        """Action mask correction works independently per agent in a multi-agent group."""
        batch, n_agents, obs_dim, n_actions = 2, 2, 4, 5
        obs = torch.zeros(batch, n_agents, obs_dim)
        mask = torch.zeros(batch, n_agents, n_actions, dtype=torch.bool)
        mask[:, 0, 2] = True  # agent 0: only action 2 valid
        mask[:, 1, 4] = True  # agent 1: only action 4 valid
        td = TensorDict({"ego": {"observation": obs, "action_mask": mask}}, batch_size=[batch])
        # Both policies want action 0, which is invalid for both agents
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0), _const_policy(0)]})
        out = router(td)
        assert (out["ego", "action"][:, 0] == 2).all()
        assert (out["ego", "action"][:, 1] == 4).all()

    def test_action_mask_not_applied_to_continuous(self):
        """Action masks must be ignored for continuous actions."""
        batch, n_agents, obs_dim, n_actions = 2, 1, 4, 5
        obs = torch.zeros(batch, n_agents, obs_dim)
        mask = torch.zeros(batch, n_agents, n_actions, dtype=torch.bool)
        td = TensorDict({"ego": {"observation": obs, "action_mask": mask}}, batch_size=[batch])
        router = MultiPolicyRouter(policies={"ego": [_const_policy(7, action_dim=3)]})
        out = router(td)
        # Continuous action should be unchanged despite all-zero mask
        assert out["ego", "action"].shape == (2, 1, 3)
        assert (out["ego", "action"] == 7.0).all()

    def test_wrong_action_batch_size_raises(self):
        td = _make_td(["ego"], batch=2, n_agents=1)

        class _WrongBatchPolicy(nn.Module):
            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                return torch.zeros(99, dtype=torch.long)

        router = MultiPolicyRouter(policies={"ego": [_WrongBatchPolicy()]})
        with pytest.raises(ValueError, match="batch size"):
            router(td)


# ---------------------------------------------------------------------------
# TestMultiPolicyRouter — RL (TensorDict) groups
# ---------------------------------------------------------------------------


class TestMultiPolicyRouterRL:
    def test_ego_only_no_teammates(self):
        td = _make_td(["ego"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": _MockEgoTDPolicy(n_ego_agents=1)})
        out = router(td)
        nested = set(out.keys(include_nested=True))
        assert ("ego", "action") in nested
        assert ("teammate", "action") not in nested

    def test_ego_and_teammates(self):
        td = _make_td(["ego", "teammate"], batch=2, n_agents=1)
        router = MultiPolicyRouter(
            policies={
                "ego": _MockEgoTDPolicy(n_ego_agents=1),
                "teammate": [_const_policy(7)],
            }
        )
        out = router(td)
        assert out["ego", "action"].shape == (2, 1)
        assert out["teammate", "action"].shape == (2, 1)
        assert (out["ego", "action"] == 0).all()
        assert (out["teammate", "action"] == 7).all()

    def test_rl_policy_returns_wrong_type_raises(self):
        class _BadRLPolicy(TensorDictModuleBase):
            in_keys = [("ego", "observation")]
            out_keys = [("ego", "action")]

            def forward(self, input_td):
                return "not a tensordict"

        td = _make_td(["ego"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": _BadRLPolicy()})
        with pytest.raises(TypeError, match="TensorDictBase"):
            router(td)

    def test_ego_policy_missing_action_raises(self):
        td = _make_td(["ego"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": _MockEgoTDPolicy(n_ego_agents=1, write_action=False)})
        with pytest.raises(KeyError):
            router(td)

    def test_observations_routed_to_correct_policy(self):
        """Ego policy receives the full TensorDict; teammate policy receives obs slice."""
        batch, obs_dim = 2, 4

        class _RecordingEgoPolicy(TensorDictModuleBase):
            in_keys = [("ego", "observation")]
            out_keys = [("ego", "action")]

            def __init__(self):
                super().__init__()
                self.received: torch.Tensor | None = None

            def forward(self, input_td: TensorDict) -> TensorDict:
                self.received = input_td["ego", "observation"].clone()
                action = torch.zeros(batch, 1, dtype=torch.long)
                input_td.set(("ego", "action"), action)
                return input_td

        class _RecordingTeammatePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.received: torch.Tensor | None = None

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                self.received = obs.clone()
                return torch.zeros(obs.shape[0], dtype=torch.long)

        ego_obs = torch.ones(batch, 1, obs_dim) * 1.0
        teammate_obs = torch.ones(batch, 1, obs_dim) * 2.0
        td = TensorDict(
            {"ego": {"observation": ego_obs}, "teammate": {"observation": teammate_obs}},
            batch_size=[batch],
        )

        ego_policy = _RecordingEgoPolicy()
        teammate_policy = _RecordingTeammatePolicy()
        router = MultiPolicyRouter(policies={"ego": ego_policy, "teammate": [teammate_policy]})
        router(td)

        assert torch.allclose(ego_policy.received, ego_obs)
        assert torch.allclose(teammate_policy.received, teammate_obs[:, 0, :])

    def test_gru_recurrent_state_passed_through(self):
        """Full TensorDict is passed to the RL policy, so recurrent state keys
        must be accessible and any updated state is preserved in the output."""
        batch, n_agents, obs_dim = 2, 1, 4
        n_layers, hidden_size = 1, 8

        class _GRULikeEgoPolicy(TensorDictModuleBase):
            in_keys = [("ego", "observation"), "is_init", ("ego", "_hidden_gru_0")]
            out_keys = [("ego", "action"), ("next", "ego", "_hidden_gru_0")]

            def __init__(self):
                super().__init__()
                self.received_is_init: torch.Tensor | None = None
                self.received_hidden: torch.Tensor | None = None

            def forward(self, input_td: TensorDict) -> TensorDict:
                self.received_is_init = input_td.get("is_init").clone()
                self.received_hidden = input_td.get(("ego", "_hidden_gru_0")).clone()
                updated_hidden = self.received_hidden + 1.0
                input_td.set(("ego", "action"), torch.zeros(batch, n_agents, dtype=torch.long))
                input_td.set(("next", "ego", "_hidden_gru_0"), updated_hidden)
                return input_td

        h0 = torch.zeros(batch, n_agents, n_layers, hidden_size)
        is_init = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            {
                "ego": {"observation": torch.zeros(batch, n_agents, obs_dim), "_hidden_gru_0": h0},
                "is_init": is_init,
            },
            batch_size=[batch],
        )

        ego_policy = _GRULikeEgoPolicy()
        router = MultiPolicyRouter(policies={"ego": ego_policy})
        out = router(td)

        assert torch.allclose(ego_policy.received_is_init, is_init)
        assert torch.allclose(ego_policy.received_hidden, h0)
        assert ("next", "ego", "_hidden_gru_0") in set(out.keys(include_nested=True))
        assert torch.allclose(out["next", "ego", "_hidden_gru_0"], h0 + 1.0)


# ---------------------------------------------------------------------------
# TestMultiPolicyRouter — construction and group resolution
# ---------------------------------------------------------------------------


class TestMultiPolicyRouterConstruction:
    def test_empty_policies_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            MultiPolicyRouter(policies={})

    def test_empty_policy_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            MultiPolicyRouter(policies={"ego": []})

    def test_invalid_policy_type_raises(self):
        with pytest.raises(TypeError):
            MultiPolicyRouter(policies={"ego": "not a module"})

    def test_invalid_list_element_raises(self):
        with pytest.raises(TypeError, match="nn.Module"):
            MultiPolicyRouter(policies={"ego": [42]})

    def test_group_not_in_td_is_skipped(self):
        """Groups in policies but absent from TD are silently skipped."""
        td = _make_td(["ego"], batch=2, n_agents=1)
        router = MultiPolicyRouter(
            policies={
                "ego": [_const_policy(0)],
                "teammate": [_const_policy(1)],
            }
        )
        out = router(td)
        nested = set(out.keys(include_nested=True))
        assert ("ego", "action") in nested
        assert ("teammate", "action") not in nested

    def test_group_in_td_but_not_in_policies_is_ignored(self):
        """Groups in TD but absent from policies are ignored."""
        td = _make_td(["ego", "teammate"], batch=2, n_agents=1)
        router = MultiPolicyRouter(policies={"ego": [_const_policy(0)]})
        out = router(td)
        nested = set(out.keys(include_nested=True))
        assert ("ego", "action") in nested
        assert ("teammate", "action") not in nested

    def test_parameters_registered(self):
        """All sub-module parameters are reachable via router.parameters()."""
        policy_with_params = nn.Linear(4, 1)
        router = MultiPolicyRouter(policies={"ego": [policy_with_params]})
        param_list = list(router.parameters())
        assert len(param_list) > 0
        assert any(p is policy_with_params.weight for p in param_list)

    def test_rl_parameters_registered(self):
        """RL policy parameters are reachable via router.parameters()."""
        ego = _MockEgoTDPolicy(n_ego_agents=1)
        router = MultiPolicyRouter(policies={"ego": ego})
        # _MockEgoTDPolicy has no parameters, but the module itself is registered
        assert any(m is ego for m in router.modules())

    def test_custom_group_names(self):
        """Router supports arbitrary group names, not just ego/teammate."""
        td = TensorDict(
            {"alpha": {"observation": torch.zeros(2, 1, 4)}, "beta": {"observation": torch.zeros(2, 1, 4)}},
            batch_size=[2],
        )
        router = MultiPolicyRouter(policies={"alpha": [_const_policy(3)], "beta": [_const_policy(5)]})
        out = router(td)
        assert (out["alpha", "action"] == 3).all()
        assert (out["beta", "action"] == 5).all()
