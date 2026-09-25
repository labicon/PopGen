"""Self-play policy loading and adaptation utilities for cooperator evaluation."""

from __future__ import annotations

import logging
import pickle
import sys
import tempfile
import types
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict


def _make_reload_save_folder(experiment_folder: Path) -> str:
    """Return an isolated scratch directory for BenchMARL experiment reloads."""
    return tempfile.mkdtemp(prefix=f"reload_{experiment_folder.name}_")


def _reload_experiment_with_callback_fallback(Experiment, checkpoint: Path, experiment_patch: dict[str, object]):
    """Reload a BenchMARL experiment, tolerating a corrupted callbacks tail in config.pkl."""
    try:
        return Experiment.reload_from_file(str(checkpoint), experiment_patch=experiment_patch)
    except (pickle.UnpicklingError, EOFError) as exc:
        config_file = checkpoint.parent.parent.resolve() / "config.pkl"
        logging.warning(
            "Falling back to callback-free reload for %s after config.pkl callback load failed: %s",
            config_file,
            exc,
        )

        with open(config_file, "rb") as f:
            task = pickle.load(f)
            task_config = pickle.load(f)
            algorithm_config = pickle.load(f)
            model_config = pickle.load(f)
            seed = pickle.load(f)
            experiment_config = pickle.load(f)
            critic_model_config = pickle.load(f)
            try:
                callbacks = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as callback_exc:
                logging.warning("Ignoring corrupted callbacks in %s: %s", config_file, callback_exc)
                callbacks = []

        task.config = task_config
        experiment_config.restore_file = str(checkpoint)
        for key, value in experiment_patch.items():
            if not hasattr(experiment_config, key):
                raise ValueError(f"Experiment config does not have attribute {key}")
            setattr(experiment_config, key, value)

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks,
            critic_model_config=critic_model_config,
        )
        print(f"\nReloaded experiment {experiment.name} from {checkpoint}.")
        return experiment


class SelfPlayPolicyWrapper(nn.Module):
    """Adapts a jointly-trained N-agent self-play policy for single-agent calls.

    When called with a TensorDict containing one agent's observation under an
    arbitrary group key, expands it to N agent copies (as the self-play policy
    expects), runs the policy, and returns agent-0's action.
    """

    def __init__(
        self,
        policy: nn.Module,
        trained_group: str = "agents",
        n_agents_trained: int = 2,
        seed_dir: Path | None = None,
    ):
        super().__init__()
        self.policy = policy
        self.trained_group = trained_group
        self.n_agents_trained = n_agents_trained
        self.seed_dir = seed_dir

    def forward(self, td: TensorDict) -> TensorDict:
        call_group = next(iter(td.keys()))
        input_device = td.device

        try:
            policy_device = next(self.policy.parameters()).device
        except StopIteration:
            policy_device = input_device

        call_td = td[call_group]
        single_obs = call_td["observation"]
        single_action_mask = call_td.get("action_mask", None)
        n_batch = single_obs.shape[0]
        obs_dim = single_obs.shape[-1]

        remapped: dict = {"observation": single_obs.expand(n_batch, self.n_agents_trained, obs_dim).contiguous().to(policy_device)}
        if single_action_mask is not None:
            action_dim = single_action_mask.shape[-1]
            remapped["action_mask"] = single_action_mask.expand(n_batch, self.n_agents_trained, action_dim).contiguous().to(policy_device)

        remapped_in = TensorDict(
            {
                self.trained_group: TensorDict(
                    remapped,
                    batch_size=[n_batch, self.n_agents_trained],
                    device=policy_device,
                )
            },
            batch_size=[n_batch],
            device=policy_device,
        )

        with torch.no_grad():
            out = self.policy(remapped_in)

        out_single = out[self.trained_group][:, :1].to(input_device)
        return TensorDict({call_group: out_single}, batch_size=[n_batch], device=input_device)


class SelfPlayPolicyTensorAdapter(nn.Module):
    """Tensor-only interface for SelfPlayPolicyWrapper.

    Converts a ``(batch, obs_dim)`` tensor call into the TensorDict interface
    expected by :class:`SelfPlayPolicyWrapper` and returns a flat action tensor,
    making self-play policies compatible with ``MultiPolicyRouter``'s
    ``list[nn.Module]`` policy format.
    """

    def __init__(self, wrapper: SelfPlayPolicyWrapper):
        super().__init__()
        self._wrapper = wrapper

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_batch = obs.shape[0]
        agent_td: dict = {"observation": obs.unsqueeze(1)}
        if action_mask is not None:
            agent_td["action_mask"] = action_mask.unsqueeze(1)
        td = TensorDict(
            {"_sp": TensorDict(agent_td, batch_size=[n_batch, 1])},
            batch_size=[n_batch],
            device=obs.device,
        )
        td_out = self._wrapper(td)
        return td_out["_sp", "action"][:, 0]


def load_selfplay_policy(
    seed_dir: str | Path,
    reload_context: Callable | None = None,
) -> SelfPlayPolicyWrapper:
    """Load the latest checkpoint from a self-play seed directory.

    Returns a :class:`SelfPlayPolicyWrapper` with the policy moved to CPU,
    ready to be wrapped by :class:`SelfPlayPolicyTensorAdapter` for use in
    ``MultiPolicyRouter``.
    """
    from benchmarl.experiment import Experiment

    seed_dir = Path(seed_dir)
    logging.info("Loading self-play policy from %s", seed_dir)

    checkpoints = sorted(
        seed_dir.rglob("checkpoints/checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {seed_dir}")
    checkpoint = checkpoints[-1]
    experiment_folder = checkpoint.parent.parent.resolve()

    _StubUnpickler = _make_stub_unpickler()
    original_pickle_load = pickle.load

    def _patched_load(f, **kwargs):
        return _StubUnpickler(f).load()

    ctx = reload_context() if callable(reload_context) else nullcontext()
    with ctx:
        pickle.load = _patched_load
        try:
            experiment_patch = {
                # BenchMARL rewrites config.pkl during reload; keep that away from the source artifact.
                "save_folder": _make_reload_save_folder(experiment_folder),
                "loggers": [],
                "create_json": False,
                "sampling_device": "cpu",
                "train_device": "cpu",
                "buffer_device": "cpu",
                "restore_map_location": "cpu",
                "parallel_collection": False,
                "collect_with_grad": True,
                "evaluation": False,
                "render": False,
                "evaluation_episodes": 1,
                "on_policy_n_envs_per_worker": 1,
                "off_policy_n_envs_per_worker": 1,
                "on_policy_collected_frames_per_batch": 1,
                "off_policy_collected_frames_per_batch": 1,
            }
            experiment = _reload_experiment_with_callback_fallback(
                Experiment,
                checkpoint,
                experiment_patch,
            )
        finally:
            pickle.load = original_pickle_load

    trained_group = "ego" if "ego" in experiment.group_policies else "agents"
    if trained_group not in experiment.group_policies:
        trained_group = next(iter(experiment.group_policies))
        logging.warning(
            "load_selfplay_policy: no 'ego'/'agents' group in %s; using %r",
            checkpoint,
            trained_group,
        )

    env = experiment.test_env
    n_agents = _get_n_agents_from_env(env, trained_group)
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    policy = experiment.group_policies[trained_group].cpu()
    logging.info("  trained_group=%r  n_agents_trained=%d", trained_group, n_agents)
    return SelfPlayPolicyWrapper(policy, trained_group=trained_group, n_agents_trained=n_agents, seed_dir=seed_dir)


def enumerate_selfplay_seed_dirs(results_dir: str | Path) -> list[Path]:
    """Return sorted list of ``seed_*/`` directories under ``results_dir``."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Self-play results directory does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Self-play results path is not a directory: {results_dir}")
    seed_dirs = sorted(results_dir.glob("seed_*/"), key=lambda p: p.name)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_*/ subdirectories found in {results_dir}")
    return seed_dirs


def close_env_safely(env) -> None:
    """Close an environment, swallowing any exception."""
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_n_agents_from_env(env, group: str) -> int:
    try:
        return env.observation_spec[group, "observation"].shape[-2]
    except (KeyError, IndexError, AttributeError):
        pass
    for attr in ("n_agents", "num_agents"):
        if hasattr(env, attr):
            return int(getattr(env, attr))
    raise RuntimeError(f"Could not determine n_agents from env {type(env).__name__}")


def _make_stub_unpickler():
    """Return a ``pickle.Unpickler`` subclass that stubs missing ``popgen._dynamic`` modules."""
    if "popgen._dynamic" not in sys.modules:
        pkg = types.ModuleType("popgen._dynamic")
        pkg.__package__ = "popgen._dynamic"
        pkg.__path__ = []
        sys.modules["popgen._dynamic"] = pkg

    class _StubCallable:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Stub callable: original dynamic module not available")

    class _StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("popgen._dynamic."):
                if module not in sys.modules:
                    stub = types.ModuleType(module)
                    stub.__package__ = "popgen._dynamic"
                    sys.modules[module] = stub
                mod = sys.modules[module]
                if not hasattr(mod, name):
                    setattr(mod, name, _StubCallable())
                return getattr(mod, name)
            return super().find_class(module, name)

    return _StubUnpickler
