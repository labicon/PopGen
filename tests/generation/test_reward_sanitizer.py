"""Tests for the runtime guard applied to generated reward functions."""

import pickle

import torch

from popgen.utils.rewards import DEFAULT_REWARD_CLAMP, sanitize_reward_fn


def _tensor_reward(value):
    def compute_reward(self, agent):
        del self, agent
        return torch.as_tensor(value, dtype=torch.float32)

    return compute_reward


def _scalar_reward(value):
    def compute_reward(self, agent):
        del self, agent
        return value

    return compute_reward


def test_tensor_nan_and_inf_are_replaced():
    fn = sanitize_reward_fn(_tensor_reward([float("nan"), float("inf"), float("-inf")]))
    out = fn(None, None)
    assert torch.isfinite(out).all()
    assert out[0].item() == 0.0
    assert out[1].item() == DEFAULT_REWARD_CLAMP
    assert out[2].item() == -DEFAULT_REWARD_CLAMP


def test_tensor_magnitude_is_bounded():
    fn = sanitize_reward_fn(_tensor_reward([1e9, -1e9, 3.0]), clamp=100.0)
    out = fn(None, None)
    assert out.tolist() == [100.0, -100.0, 3.0]


def test_tensor_shape_dtype_and_device_are_preserved():
    fn = sanitize_reward_fn(_tensor_reward([[1.0, 2.0], [3.0, 4.0]]))
    raw = _tensor_reward([[1.0, 2.0], [3.0, 4.0]])(None, None)
    out = fn(None, None)
    assert out.shape == raw.shape
    assert out.dtype == raw.dtype
    assert out.device == raw.device


def test_in_range_rewards_are_untouched():
    """The common case must pass through numerically unchanged."""
    fn = sanitize_reward_fn(_tensor_reward([0.0, -2.5, 7.25]))
    assert fn(None, None).tolist() == [0.0, -2.5, 7.25]


def test_scalar_rewards_are_sanitized():
    assert sanitize_reward_fn(_scalar_reward(float("nan")))(None, None) == 0.0
    assert sanitize_reward_fn(_scalar_reward(1e9), clamp=100.0)(None, None) == 100.0
    assert sanitize_reward_fn(_scalar_reward(-1e9), clamp=100.0)(None, None) == -100.0
    assert sanitize_reward_fn(_scalar_reward(4.5))(None, None) == 4.5


def test_none_clamp_disables_wrapping():
    raw = _scalar_reward(1e9)
    assert sanitize_reward_fn(raw, clamp=None) is raw


def test_sanitized_reward_is_picklable():
    """BenchMARL may ship the reward function to spawned workers, so the guard
    must not turn it into an unpicklable closure."""
    fn = sanitize_reward_fn(_module_level_reward)
    restored = pickle.loads(pickle.dumps(fn))
    assert restored(None, None) == DEFAULT_REWARD_CLAMP


def _module_level_reward(self, agent):
    del self, agent
    return 1e9
