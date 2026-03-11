"""Helpers for reloading RL reward policies for rollout evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import vmas
from benchmarl.experiment import Experiment

from popgen.agents.scripted_partner import MultiPolicyWrapper
from popgen.envs.pp import semdiv_simple_tag
from popgen.utils.filesystem import find_reward_checkpoints


def latest_reward_training_log(save_dir: str | Path) -> Path | None:
    """Best-effort path to the latest BenchMARL wandb output log."""
    logs = sorted(Path(save_dir).rglob("wandb/run-*/files/output.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    return logs[-1]


@contextmanager
def _reload_vmas_scenarios():
    """Reload VMAS RewardFn checkpoints with the same custom scenarios used in training."""
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


def load_reward_rollout_artifacts(save_dir: str | Path, scripted_policy_fn=None):
    """Reload a trained RewardFn policy and its rollout env from a checkpoint."""
    save_dir = Path(save_dir)
    checkpoints = find_reward_checkpoints(save_dir)
    if not checkpoints:
        latest_log = latest_reward_training_log(save_dir)
        hint = f" Latest training log: {latest_log}" if latest_log is not None else ""
        raise FileNotFoundError(f"No checkpoints found in {save_dir}.{hint}")

    checkpoint = checkpoints[-1]
    experiment_folder = checkpoint.parent.parent.resolve()
    with _reload_vmas_scenarios():
        experiment = Experiment.reload_from_file(
            str(checkpoint),
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

    ego_group = "ego" if "ego" in experiment.group_policies else "agents"
    ego_policy = experiment.group_policies[ego_group]
    try:
        ego_policy = ego_policy.to("cpu")
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
        policies = [ego_policy, *teammate_policy]
    else:
        policies = [ego_policy, teammate_policy]
    return env, MultiPolicyWrapper(policies=policies)
