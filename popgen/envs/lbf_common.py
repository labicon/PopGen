import inspect
import os
from collections.abc import Callable
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments.lbf.common import LbfGymWrapper, LbfTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from popgen.envs.util import apply_rollout_seed, compose_rollout_policy, parse_single_scripted_policy
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, TransformedEnv


def _infer_num_agents(env: gymnasium.Env) -> int:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(base_env, "n_agents"):
        return int(base_env.n_agents)
    if hasattr(base_env, "players"):
        return len(base_env.players)
    if isinstance(env.observation_space, gymnasium.spaces.Tuple):
        return len(env.observation_space.spaces)
    raise ValueError(f"Could not infer number of agents from env type {type(env).__name__}.")


def _build_lbf_train_group_map(
    task_config: dict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build LBF training group map separating scripted teammate from learning agents.

    The first agent is placed in the trainable ``"ego"`` group; the remaining
    agents form the scripted ``"teammate"`` group.

    Returns:
        A ``(group_map, trainable_group)`` tuple where:
        - *group_map* – ``{"ego": [first_agent], "teammate": rest}``
        - *trainable_group* – ``{"ego": [first_agent]}``
    """
    n_agents = task_config.get("players")
    if n_agents is None:
        raise ValueError("Could not infer number of agents from LBF task config (expected 'players' key).")
    if n_agents < 2:
        raise ValueError(f"Need at least 2 agents to form ego/teammate split; got {n_agents}.")
    agent_names = [f"agent_{i}" for i in range(n_agents)]
    agents_group = [agent_names[0]]
    teammate_group = agent_names[1:]
    group_map = {"ego": agents_group, "teammate": teammate_group}
    trainable_group = {"ego": agents_group}
    return group_map, trainable_group


def _to_rollout_env(env, device: str = "cpu") -> EnvBase:
    if isinstance(env, TransformedEnv):
        return env
    if isinstance(env, EnvBase):
        return env
    if isinstance(env, gymnasium.Env):
        num_agents = _infer_num_agents(env)
        return LbfGymWrapper(env=env, num_agents=num_agents, device=device)
    raise TypeError(f"Unsupported env type for rollout_lbf_env: {type(env).__name__}")


def _resolve_state_env(env):
    current = env
    visited = set()
    while True:
        if isinstance(current, gymnasium.Env):
            return current
        next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "base_env", None)
        if next_env is None or id(next_env) in visited:
            return env
        visited.add(id(next_env))
        current = next_env


def rollout_lbf_env(
    policy,
    env,
    get_state_fn: Callable | None = None,
    max_steps: int = 50,
    **kwargs,
):
    """Rollout a policy input that may be a single policy or a list of policies."""
    policy = compose_rollout_policy(policy)
    rollout_env = _to_rollout_env(env)
    state_env = _resolve_state_env(rollout_env)
    policy.eval()

    def rollout_policy(tensordict):
        out = policy(tensordict)
        if not isinstance(out, TensorDictBase):
            raise TypeError(
                f"Policy output must be TensorDictBase, got {type(out).__name__}."
            )
        return out

    apply_rollout_seed(rollout_env)

    states = []
    images = []

    # Enable headless rendering for pyglet (used by lbforaging)
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    if hasattr(state_env, "render_mode"):
        try:
            state_env.render_mode = "rgb_array"
        except AttributeError:
            if hasattr(state_env, "unwrapped") and hasattr(state_env.unwrapped, "render_mode"):
                state_env.unwrapped.render_mode = "rgb_array"

    def render_callback(env, td):
        try:
            frame = env.render(mode="rgb_array")
        except TypeError:
            frame = env.render()
        if isinstance(frame, np.ndarray):
            images.append(frame)
        if get_state_fn is not None:
            states.append(get_state_fn(state_env))

    # Filter kwargs to only pass supported arguments to env.rollout
    rollout_sig = inspect.signature(rollout_env.rollout)
    supported_args = set(rollout_sig.parameters.keys())
    rollout_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}
    if "max_steps" in supported_args and "max_steps" not in rollout_kwargs:
        rollout_kwargs["max_steps"] = max_steps

    with torch.no_grad():
        rollout_td = rollout_env.rollout(
            policy=rollout_policy,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_any_done=True,
            **rollout_kwargs,
        )

    # Extract states (obs) if custom state function is not provided
    if get_state_fn is None:
        for step_idx in range(rollout_td.batch_size[0]):
            step_td = rollout_td[step_idx]
            state = {}
            if ("agents", "observation") in step_td.keys(include_nested=True):
                obs = step_td["agents", "observation"]
                state["observation"] = obs.cpu().numpy().tolist()
            states.append(state)

    return states, images, None

def train_lbf_env(
    reward_fn,
    output_dir,
    rl_cfg_dir,
    task_yaml_name="lbf.yaml",
    mappo_yaml_name="mappo.yaml",
    experiment_yaml_name="base_experiment.yaml",
    mlp_yaml_name="mlp.yaml",
    seed=0,
    callbacks=None,
    teammate_policy: list[nn.Module] | None = None,
):
    # Ensure reward_fn has access to torch
    if hasattr(reward_fn, "__globals__"):
        reward_fn.__globals__["torch"] = torch

    # Enable headless rendering for pyglet (used by lbforaging)
    os.environ.setdefault("PYGLET_HEADLESS", "true")

    callbacks = list(callbacks or [])

    original_step = LbfGymWrapper._step

    try:
        # Load Configs
        task = LbfTask.LBF.get_from_yaml(path=str(rl_cfg_dir / task_yaml_name))

        if teammate_policy is not None:
            group_map, trainable_group = _build_lbf_train_group_map(task_config=task.config)
            task.config["group_map"] = group_map
            from popgen.agents.scripted_partner import TeammatePolicyCallback

            callbacks.append(
                TeammatePolicyCallback(
                    teammate_policies=teammate_policy,
                    trainable_group=trainable_group,
                )
            )

        desired_group_map = task.config.get("group_map")

        if desired_group_map:
            name_to_group = {
                aname: (gname, local_i)
                for gname, names in desired_group_map.items()
                for local_i, aname in enumerate(names)
            }

        def custom_step(self, tensordict):
            out_td = original_step(self, tensordict)
            base_env = self.env.unwrapped

            new_rewards = []
            for i, player in enumerate(base_env.players):
                if desired_group_map:
                    gname, local_i = name_to_group[f"agent_{i}"]
                    player.obs = out_td[gname, "observation"][local_i].cpu().numpy()
                else:
                    player.obs = out_td["agents", "observation"][i].cpu().numpy()
                try:
                    r = reward_fn(base_env, player)
                    if isinstance(r, torch.Tensor):
                        r = r.item()
                    new_rewards.append(r)
                except Exception as e:
                    raise e

            if desired_group_map:
                for gname, names in desired_group_map.items():
                    idxs = [int(n.split("_")[1]) for n in names]
                    rt = torch.tensor(
                        [new_rewards[idx] for idx in idxs],
                        device=self.device,
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                    out_td[gname, "reward"] = rt
            else:
                reward_tensor = torch.tensor(
                    new_rewards, device=self.device, dtype=torch.float32
                ).unsqueeze(-1)
                out_td["agents", "reward"] = reward_tensor

            return out_td

        LbfGymWrapper._step = custom_step

        mappo_path = rl_cfg_dir / mappo_yaml_name
        if mappo_path.exists():
            algorithm_config = MappoConfig.get_from_yaml(path=str(mappo_path))
        elif (rl_cfg_dir / "ippo.yaml").exists():
            algorithm_config = IppoConfig.get_from_yaml(path=str(rl_cfg_dir / "ippo.yaml"))
        else:
            algorithm_config = MappoConfig.get_from_yaml()

        experiment_config = ExperimentConfig.get_from_yaml(path=str(rl_cfg_dir / experiment_yaml_name))
        model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / mlp_yaml_name))
        critic_model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / mlp_yaml_name))

        # Override experiment config for output
        experiment_config.save_folder = output_dir

        # Name run
        run_name = Path(output_dir).name
        experiment_config.wandb_extra_kwargs = {"name": run_name, "tags": ["lbf"]}

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks,
        )

        experiment.run()

    finally:
        # Restore original _step
        LbfGymWrapper._step = original_step

    return experiment
