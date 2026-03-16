"""Policy code agent wrapper for deterministic action functions."""

import torch
import torch.nn as nn

from .utils import extract_observation_and_action_key, flatten_obs, format_action_output


class PolicyCodeAgent(nn.Module):
    """Takes a deterministic function from observations to actions."""

    def __init__(self, act_fn: callable):
        super().__init__()
        self.act_fn = act_fn

    def _normalize_action(self, action, obs_dtype: torch.dtype, obs_device: torch.device) -> torch.Tensor:
        """Coerce a single-slot action from act_fn into a scalar or 1-D torch.Tensor.

        act_fn is LLM-generated and may return a Python scalar (int/float),
        a list, a NumPy array, or a torch.Tensor of any rank.  This method
        normalises all of those cases so that torch.stack in the caller
        always receives a uniform 1-D tensor.

        Args:
            action: Raw output of act_fn for one observation slot.
            obs_dtype: dtype of the observation tensor; used as the target
                dtype so that action and observation live in the same numeric
                type (e.g. both float32).
            obs_device: device of the observation tensor; action is moved
                here to avoid cross-device stack errors.

        Returns:
            A scalar tensor for discrete actions or a 1-D tensor for vector actions,
            always on obs_device.

        Raises:
            ValueError: If action has more than one element along its first
                axis (i.e. act_fn returned a multi-slot tensor).
        """
        if not isinstance(action, torch.Tensor):
            # Handles Python scalars, lists, NumPy arrays, etc.
            # torch.as_tensor shares memory when possible and always applies
            # the requested dtype and device in one step.
            action = torch.as_tensor(action, dtype=obs_dtype, device=obs_device)
        else:
            # NOTE: only the device is moved here; dtype is not enforced for
            # tensor inputs. If act_fn returns a float64 tensor but obs_dtype
            # is float32, the mismatch will surface at torch.stack.
            action = action.to(device=obs_device)

        if action.dim() > 1:
            # act_fn returned a 2-D+ tensor.  The only tolerated case is a
            # spurious leading batch dimension of size 1 (e.g. shape (1, k)),
            # which we strip.  Anything else indicates that act_fn computed
            # actions for multiple slots at once, which is not supported here.
            if action.shape[0] != 1:
                raise ValueError("PolicyCodeAgent expects act_fn to return a single action per observation slot.")
            action = action.squeeze(0)

        return action

    def forward(self, input_td):
        """Forward pass with TensorDict-in/TensorDict-out and tensor fallback."""
        obs_tensor, action_key = extract_observation_and_action_key(input_td)
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(f"PolicyCodeAgent expected torch.Tensor observation, got {type(obs_tensor).__name__}.")
        if obs_tensor.dim() == 0:
            raise ValueError("PolicyCodeAgent expects observation tensor with at least 1 dimension.")

        flat_obs, leading_shape = flatten_obs(obs_tensor)

        actions = []
        for slot_obs in flat_obs:
            try:
                slot_action = self.act_fn(slot_obs)
            except RuntimeError as exc:
                # Generated policies sometimes construct CPU tensors explicitly
                # (for example torch.tensor([...])) while obs is on CUDA.
                # Retry on CPU and move result back in _normalize_action.
                if "same device" not in str(exc):
                    raise
                slot_action = self.act_fn(slot_obs.cpu())
            slot_action = self._normalize_action(
                slot_action,
                obs_dtype=obs_tensor.dtype,
                obs_device=obs_tensor.device,
            )
            actions.append(slot_action)

        action_flat = torch.stack(actions)
        if action_flat.dim() == 1:
            # Discrete scalar actions should keep shape (...), not (..., 1).
            action_tensor = action_flat[0] if len(leading_shape) == 0 else action_flat.reshape(*leading_shape)
        else:
            action_tensor = action_flat.reshape(*leading_shape, action_flat.shape[-1])

        return format_action_output(input_td, action_key, action_tensor)
