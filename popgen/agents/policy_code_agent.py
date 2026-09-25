"""Policy code agent wrapper for deterministic action functions."""

import torch
import torch.nn as nn


class PolicyCodeAgent(nn.Module):
    """Takes a deterministic function from observations to actions."""

    def __init__(self, act_fn: callable):
        super().__init__()
        self.act_fn = act_fn
        self._vmap_compatible: bool | None = None  # None=untested, True/False memoized

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

    def _call_single(self, slot_obs: torch.Tensor) -> torch.Tensor:
        """Call act_fn on one observation with device-mismatch fallback."""
        try:
            slot_action = self.act_fn(slot_obs)
        except RuntimeError as exc:
            if "same device" not in str(exc):
                raise
            slot_action = self.act_fn(slot_obs.cpu())
        return self._normalize_action(slot_action, obs_dtype=slot_obs.dtype, obs_device=slot_obs.device)

    def _purge_vmap_state(self) -> None:
        """Delete all function attributes left on act_fn by a previous vmap call.

        When vmap runs act_fn and the function writes output tensors to its own
        attributes (e.g. ``act_fn.prev_obs = x``), those attributes end up holding
        *batched* tensors tied to the vmap dispatch level.  Accessing them outside
        that context raises "tensor may have escaped from inside a function being
        vmapped".

        Deleting (not just setting to None) all attributes forces any ``hasattr``
        guards in the LLM-generated policy to re-run their initialization block,
        so the function returns to a fully clean initial state.

        The policy loses accumulated history, but resumes correct operation
        within a few timesteps — far preferable to crashing the run.
        """
        for key in list(getattr(self.act_fn, "__dict__", {}).keys()):
            try:
                delattr(self.act_fn, key)
            except AttributeError:
                pass

    def _forward_loop(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Sequential per-slot fallback for vmap-incompatible act_fn."""
        try:
            actions = [self._call_single(slot_obs) for slot_obs in obs_tensor]
        except RuntimeError as exc:
            if "escaped" not in str(exc) and "vmapped" not in str(exc) and "data-dependent" not in str(exc):
                raise
            # act_fn's internal state holds batched tensors from a previous
            # successful vmap call.  Reset them and retry the loop.
            self._purge_vmap_state()
            actions = [self._call_single(slot_obs) for slot_obs in obs_tensor]
        action_flat = torch.stack(actions)
        if action_flat.dim() == 1:
            return action_flat
        return action_flat.reshape(obs_tensor.shape[0], action_flat.shape[-1])

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass on an observation tensor."""
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(f"PolicyCodeAgent expected torch.Tensor observation, got {type(obs_tensor).__name__}.")
        if obs_tensor.dim() != 2:
            raise ValueError(f"PolicyCodeAgent expects observation tensor with shape (batch, obs_dim), got {tuple(obs_tensor.shape)}.")

        if obs_tensor.shape[0] == 1:
            # Batch=1 fast path: skip vmap overhead entirely.
            return self._call_single(obs_tensor[0]).unsqueeze(0)

        if self._vmap_compatible is False:
            return self._forward_loop(obs_tensor)

        try:
            action_tensor = torch.vmap(self.act_fn)(obs_tensor)
            self._vmap_compatible = True
            if action_tensor.dim() == 1:
                return action_tensor
            return action_tensor.reshape(obs_tensor.shape[0], -1)
        except Exception:
            # Memoize so future calls skip re-tracing.
            self._vmap_compatible = False
            return self._forward_loop(obs_tensor)
