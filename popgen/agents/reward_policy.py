"""Helpers for reloading RL reward policies for rollout evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import vmas
from benchmarl.experiment import Experiment
from tensordict import TensorDict

from popgen.agents.router import MultiPolicyRouter
from popgen.envs.pp import semdiv_simple_tag
from popgen.utils.benchmarl_reload import reload_experiment
from popgen.utils.filesystem import find_reward_checkpoints


def latest_reward_training_log(save_dir: str | Path) -> Path | None:
    """Best-effort path to the latest BenchMARL wandb output log."""
    logs = sorted(Path(save_dir).rglob("wandb/run-*/files/output.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    return logs[-1]


@contextmanager
def _reload_scenarios(reload_context=None):
    """Rebuild the checkpoint's model under the scenario it was trained with.

    A checkpoint can only be reloaded into a model with the same observation
    shape, so the scenario active during reload must match the one active
    during training. Environments that patch the VMAS scenario expose
    ``get_cooperator_reload_context`` for exactly this; prefer it.

    Without it, `pp` and `pp_stag` -- which both register the VMAS scenario
    name "simple_tag" but ship *different* `semdiv_simple_tag` modules -- both
    fell back to `pp`'s scenario. For pp_stag that produced a 30- vs 36-wide
    observation and the reload died with "Failed to update 'weight'", after the
    full training had already run.
    """
    if reload_context is not None:
        with reload_context():
            yield
        return

    original_make_env = vmas.make_env

    def custom_make_env(scenario, *args, **kwargs):
        if scenario == "simple_tag":
            return original_make_env(semdiv_simple_tag.Scenario(), *args, **kwargs)
        return original_make_env(scenario, *args, **kwargs)

    vmas.make_env = custom_make_env
    try:
        yield
    finally:
        vmas.make_env = original_make_env


def load_reward_rollout_artifacts(save_dir: str | Path, scripted_policy_fn=None, reload_context=None, device: str = "cpu"):
    """Reload a trained RewardFn policy and its rollout env from a checkpoint.

    ``device`` is where the reloaded policy is placed. It defaults to "cpu",
    which is right for evaluation rollouts (one environment, and the rollout
    workers are spawned processes where initialising CUDA is best avoided).

    Cooperator training must override it. There the teammate runs once per
    environment step against a large batch of parallel envs, and on CPU that
    forward pass dominates the iteration: measured at 90s of pp_stag's 120s
    iteration (600 envs) and 284s for overcooked (400 steps/iteration), versus
    ~1s on CUDA. Leaving teammates on CPU also forces a GPU->CPU->GPU round
    trip per step, which is why the GPU sat at 0-2% utilisation.
    """
    save_dir = Path(save_dir)
    checkpoints = find_reward_checkpoints(save_dir)
    if not checkpoints:
        latest_log = latest_reward_training_log(save_dir)
        hint = f" Latest training log: {latest_log}" if latest_log is not None else ""
        raise FileNotFoundError(f"No checkpoints found in {save_dir}.{hint}")

    checkpoint = checkpoints[-1]
    experiment_folder = checkpoint.parent.parent.resolve()
    with _reload_scenarios(reload_context):
        # Via reload_experiment, not Experiment.reload_from_file: a member trained by a
        # Hydra script pickles callbacks from a module that is only importable with that
        # script's directory on sys.path (e.g. MEP's `callback`). Callbacks are
        # training-only, so they are dropped rather than reproduced.
        experiment = reload_experiment(
            Experiment,
            checkpoint,
            experiment_patch={
                "save_folder": str(experiment_folder.parent),
                "loggers": [],
                "create_json": False,
                "evaluation": False,
                "evaluation_episodes": 1,
                "sampling_device": "cpu",
                "train_device": "cpu",
                "buffer_device": "cpu",
                "restore_map_location": "cpu",
            },
        )

    if "ego" not in experiment.group_policies:
        raise KeyError(f"Expected semantic 'ego' policy group in reloaded experiment, got {list(experiment.group_policies.keys())}.")
    ego_policy = experiment.group_policies["ego"]
    try:
        ego_policy = ego_policy.to(device)
    except Exception:
        pass
    ego_policy.eval()

    env = experiment.test_env
    try:
        env = env.to("cpu")
    except Exception:
        pass

    if scripted_policy_fn is None:
        return env, ego_policy

    teammate_policy = scripted_policy_fn(env)
    if isinstance(teammate_policy, (list, tuple)):
        teammate_policies = list(teammate_policy)
    else:
        teammate_policies = [teammate_policy]
    return env, MultiPolicyRouter(policies={"ego": ego_policy, "teammate": teammate_policies})


class RewardPolicyTensorAdapter(nn.Module):
    """Tensor-only interface for a reloaded BenchMARL reward policy.

    Teammate routing calls each policy as ``policy(obs_tensor)`` with a
    ``(batch, obs_dim)`` tensor, which is what ``PolicyCodeAgent`` accepts. A
    reward-trained teammate is a BenchMARL ``ProbabilisticActor`` expecting a
    nested TensorDict with an agent dimension, so it must be adapted before it
    can be used as a teammate.

    Mirrors :class:`popgen.cooperator.selfplay_util.SelfPlayPolicyTensorAdapter`,
    which solves the same mismatch for self-play policies.
    """

    def __init__(self, policy: nn.Module, group: str = "ego"):
        super().__init__()
        self.policy = policy
        self.group = group

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        n_batch = obs.shape[0]
        # The policy was trained as a single-agent group, so add the agent dim.
        agent_td: dict = {"observation": obs.unsqueeze(1)}
        if action_mask is not None:
            agent_td["action_mask"] = action_mask.unsqueeze(1)
        td = TensorDict(
            {self.group: TensorDict(agent_td, batch_size=[n_batch, 1])},
            batch_size=[n_batch],
            device=obs.device,
        )
        with torch.no_grad():
            out = self.policy(td)
        return out[self.group, "action"][:, 0]
