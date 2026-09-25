"""The generated Overcooked reward must apply to the ego group only.

It used to be evaluated for every agent in the group map. Nothing trains on the
teammate group (``trainable_group`` is ego-only), so that work was discarded --
but it still polluted ``collection/reward/*`` (the mean over groups) and
inverted the sign of behaviours that deliberately push a task onto the partner.
"""

import torch
from tensordict import TensorDict

from popgen.envs.overcooked.utils import _build_overcooked_group_map


def _rebuild_custom_step(reward_fn, group_map, trainable_group, original_step):
    """Mirror of the reward-override wiring in train_overcooked_env.

    Kept as a local reconstruction so the test does not need a GPU, BenchMARL,
    or a live Overcooked environment.
    """
    desired_group_map = group_map
    ego_group, ego_agent_names = next(iter(trainable_group.items()))
    ego_idxs = [int(name.split("_")[1]) for name in ego_agent_names]

    def custom_step(self, tensordict):
        out_td = original_step(self, tensordict)
        ego_rewards = []
        for idx in ego_idxs:
            r = reward_fn(self.oc_env, idx)
            if isinstance(r, torch.Tensor):
                r = r.item()
            ego_rewards.append(float(r))
        reward_tensor = torch.tensor(ego_rewards, device=self.device, dtype=torch.float32).unsqueeze(-1)

        if desired_group_map:
            out_td[ego_group, "reward"] = reward_tensor
        else:
            flat = out_td["agents", "reward"].clone()
            for row, idx in enumerate(ego_idxs):
                flat[idx] = reward_tensor[row]
            out_td["agents", "reward"] = flat
        return out_td

    return custom_step


class _FakeWrapper:
    device = "cpu"
    oc_env = object()


def _grouped_env_step(_self, _td):
    return TensorDict(
        {
            "ego": TensorDict({"reward": torch.tensor([[1.0]])}, batch_size=[1]),
            "teammate": TensorDict({"reward": torch.tensor([[2.0]])}, batch_size=[1]),
        },
        batch_size=[],
    )


def test_generated_reward_applies_to_ego_group_only():
    calls = []

    def reward_fn(_env, agent_index):
        calls.append(agent_index)
        return 100.0 + agent_index

    group_map, trainable_group = _build_overcooked_group_map()
    step = _rebuild_custom_step(reward_fn, group_map, trainable_group, _grouped_env_step)
    out = step(_FakeWrapper(), None)

    assert calls == [0], "reward_fn must be evaluated for the ego agent only"
    assert out["ego", "reward"].item() == 100.0
    # The teammate keeps the environment's own reward, so collection/reward is
    # no longer half a quantity nothing trains on.
    assert out["teammate", "reward"].item() == 2.0


def test_flat_layout_preserves_other_agents_reward():
    def reward_fn(_env, agent_index):
        return 100.0 + agent_index

    def flat_env_step(_self, _td):
        return TensorDict(
            {"agents": TensorDict({"reward": torch.tensor([[1.0], [2.0]])}, batch_size=[2])},
            batch_size=[],
        )

    _, trainable_group = _build_overcooked_group_map()
    step = _rebuild_custom_step(reward_fn, None, trainable_group, flat_env_step)
    out = step(_FakeWrapper(), None)

    assert out["agents", "reward"][0].item() == 100.0
    assert out["agents", "reward"][1].item() == 2.0


def test_cooperator_path_is_untouched():
    """With reward_fn=None the override is never installed at all.

    CooperatorTrainingPipeline always passes reward_fn=None, and it is the only
    caller that attaches TeamRewardSumCallback (which folds the teammate group's
    reward into the ego's training signal). The two must not interact.
    """
    import inspect

    from popgen.cooperator.training import CooperatorTrainingPipeline
    from popgen.envs.overcooked.utils import train_overcooked_env

    assert inspect.signature(CooperatorTrainingPipeline.__init__).parameters["reward_fn"].default is None

    source = inspect.getsource(train_overcooked_env)
    assert "if reward_fn is not None:" in source, "the override must stay guarded on reward_fn"


def test_real_wiring_matches_this_reconstruction():
    """Guard against the reconstruction above drifting from the real code.

    `train_overcooked_env` needs BenchMARL and a live Overcooked env, so the
    behavioural tests rebuild its closure. Pin the two properties that make
    that rebuild faithful.
    """
    import inspect

    from popgen.envs.overcooked.utils import train_overcooked_env

    source = inspect.getsource(train_overcooked_env)
    assert "next(iter(trainable_group.items()))" in source, "ego group must be derived from trainable_group"
    assert "for gname, names in desired_group_map.items()" not in source, (
        "reward_fn must not be evaluated for every group again"
    )
