import inspect
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import gymnasium
import numpy as np
import torch
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments.lbf.common import LbfGymWrapper, LbfTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from tensordict import TensorDict


class BenchMARLLbfWrapper(gymnasium.Wrapper):
    """
    Wrapper for LBF environment to receive action and return observation in
    similar format with BenchMARL.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        obs = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(0)
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Handle (1, n_agents, 1)
        if action.ndim == 3 and action.shape[0] == 1 and action.shape[2] == 1:
            action = action.squeeze(0).squeeze(-1)
        # Handle (1, n_agents) just in case
        elif action.ndim == 2 and action.shape[0] == 1:
            action = action.squeeze(0)

        # Ensure it is integer for LBF
        action = action.astype(int)

        obs, reward, done, truncated, info = self.env.step(tuple(action))
        obs = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)

        return obs, reward, done, truncated, info


def rollout_lbf_env(policy, env, get_state_fn: Callable | None = None, max_steps: int = 50, **kwargs):
    """
    Rollout the policy in the LBF environment.

    Args:
        policy: The policy to evaluate. It receives a list of parsed observations (dicts)
                and should return a TensorDict with "agents" -> "action".
        env: The LBF environment (gymnasium.Env).
        get_state_fn: Function to extract state from env.
        max_steps: Maximum steps to run.
        **kwargs: Extra args.
    """
    if not isinstance(env, BenchMARLLbfWrapper):
        env = BenchMARLLbfWrapper(env)

    policy.eval()

    # Seed
    seed = (int(time.time() * 1000) + os.getpid()) % 100000
    obs, _ = env.reset(seed=seed)

    states = []
    images = []

    # Enable headless rendering for pyglet (used by lbforaging)
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    # Ensure the base env uses rgb_array mode regardless of how it was created
    env.unwrapped.render_mode = "rgb_array"

    for step in range(max_steps):
        frame = env.render()
        if isinstance(frame, np.ndarray):
            images.append(frame)

        # State
        if get_state_fn is not None:
            s = get_state_fn(env)
            s["step"] = step
            states.append(s)
        else:
            # Fallback: store raw observations
            state = {}
            # obs is (1, n_agents, obs_dim) tensor
            state["observation"] = obs.squeeze(0).tolist()
            state["step"] = step
            states.append(state)

        # Prepare Policy Input
        policy_input = TensorDict({"observation": obs}, batch_size=[1])

        # Policy Inference
        with torch.no_grad():
            out = policy(policy_input)

        # Extract Actions
        if isinstance(out, torch.Tensor):
            actions_tensor = out
        else:
            actions_tensor = out["agents", "action"]

        # Step Env
        obs, rewards, done, truncated, info = env.step(actions_tensor)

        if done or truncated:
            break

    return states, images, None


def train_lbf_env(reward_fn, output_dir, rl_cfg_dir, task_yaml_name="lbf.yaml", mappo_yaml_name="mappo.yaml", experiment_yaml_name="base_experiment.yaml", mlp_yaml_name="mlp.yaml", seed=0):
    # Ensure reward_fn has access to torch
    if hasattr(reward_fn, "__globals__"):
        reward_fn.__globals__["torch"] = torch

    # Monkey Patch LbfGymWrapper._step to inject custom rewards
    original_step = LbfGymWrapper._step

    def custom_step(self, tensordict):
        # 1. Run the original step (physics step + original reward calculation)
        out_td = original_step(self, tensordict)

        # 2. Calculate custom rewards
        # LbfGymWrapper has self.env which is the Gymnasium environment
        base_env = self.env.unwrapped

        new_rewards = []
        # Access players from the base env
        for i, player in enumerate(base_env.players):
            try:
                # Provide player and env to reward_fn
                r = reward_fn(player, base_env)
                if isinstance(r, torch.Tensor):
                    r = r.item()
                new_rewards.append(r)
            except Exception as e:
                # Re-raise for debugging purposes
                raise e

        # 3. Overwrite rewards in the output TensorDict
        # new_rewards is list of floats. Convert to tensor (n_agents, 1)
        reward_tensor = torch.tensor(new_rewards, device=self.device, dtype=torch.float32).unsqueeze(-1)

        out_td["agents", "reward"] = reward_tensor
        return out_td

    LbfGymWrapper._step = custom_step

    try:
        # Load Configs
        task = LbfTask.get_from_yaml(path=str(rl_cfg_dir / task_yaml_name))

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
        )

        experiment.run()

    finally:
        # Restore original _step
        LbfGymWrapper._step = original_step

    return experiment
