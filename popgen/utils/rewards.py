"""Runtime guards for LLM-generated reward functions.

Generated rewards are executed verbatim inside the training loop. An unbounded
or NaN-producing reward drives the PPO actor to diverge (`entropy`/
`loss_objective` go NaN while the critic stays healthy), which BenchMARL
surfaces only as a crash tens of minutes into training. Bounding the reward at
the source keeps that failure mode out of the training loop.

The magnitude budget advertised here is the same one stated to the model in
each environment's ``REWARD_API_REFERENCE``; keep the two in sync.
"""

import torch

DEFAULT_REWARD_CLAMP = 10.0


class _SanitizedRewardFn:
    """Wrap a reward function, replacing NaN/inf and bounding magnitude.

    Implemented as a module-level callable rather than a closure so the result
    stays picklable — generated reward functions are resolved through
    ``popgen.utils.imports``, whose module registration exists precisely so
    ``pickle`` can find them by ``__module__``.
    """

    def __init__(self, fn, clamp: float = DEFAULT_REWARD_CLAMP):
        self.fn = fn
        self.clamp = float(clamp)
        # Mirror the defensive `torch` injection the environments perform, so
        # wrapping does not deprive the generated module of it.
        fn_globals = getattr(fn, "__globals__", None)
        if isinstance(fn_globals, dict):
            fn_globals.setdefault("torch", torch)

    def __call__(self, *args, **kwargs):
        return self._sanitize(self.fn(*args, **kwargs))

    def _sanitize(self, reward):
        lo, hi = -self.clamp, self.clamp
        if isinstance(reward, torch.Tensor):
            # nan_to_num first: clamping a NaN leaves it NaN.
            return torch.nan_to_num(reward, nan=0.0, posinf=hi, neginf=lo).clamp(lo, hi)
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            # Unknown type: leave it to the environment to reject.
            return reward
        value = float(reward)
        if value != value:  # NaN
            return 0.0
        return max(lo, min(hi, value))


def sanitize_reward_fn(fn, clamp: float | None = DEFAULT_REWARD_CLAMP):
    """Return *fn* guarded against NaN/inf and out-of-range magnitudes.

    ``clamp=None`` disables sanitizing and returns *fn* unchanged.
    """
    if fn is None or clamp is None:
        return fn
    return _SanitizedRewardFn(fn, clamp)
