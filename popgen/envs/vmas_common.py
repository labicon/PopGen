import inspect
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import vmas
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


def train_vmas_env(
    reward_fn,
    output_dir,
    scenario_module,
    scenario_name,
    vmas_task,
    rl_cfg_dir,
    task_yaml_name=None,
    mappo_yaml_name="mappo.yaml",
    experiment_yaml_name="base_experiment.yaml",
    mlp_yaml_name="mlp.yaml",
):
    if task_yaml_name is None:
        task_yaml_name = f"{scenario_name}.yaml"

    # Ensure reward_fn has access to torch
    if hasattr(reward_fn, "__globals__"):
        reward_fn.__globals__["torch"] = torch

    # 1. Save original Scenario class to restore later
    OriginalScenario = scenario_module.Scenario

    # 2. Define Custom Scenario Wrapper
    class CustomScenario(OriginalScenario):
        def reward(self, agent):
            # Only override reward for the first agent
            if agent == self.world.agents[0]:
                batch_dim = self.world.batch_dim
                self.rew = reward_fn(self, agent)
                if self.rew.shape != (batch_dim,):
                    self.rew = self.rew.squeeze()
                    if self.rew.ndim == 0:
                        self.rew = self.rew.view(1)
                return self.rew
            # Use original reward for other agents
            return super().reward(agent)

    # 3. Monkey Patch: Replace make_env in vmas to use CustomScenario
    original_make_env = vmas.make_env

    def custom_make_env(scenario, *args, **kwargs):
        if scenario == scenario_name:
            # Instantiate our custom scenario
            scenario_instance = CustomScenario()
            return original_make_env(
                scenario_instance,
                *args,
                **kwargs,
            )
        return original_make_env(scenario, *args, **kwargs)

    vmas.make_env = custom_make_env

    experiment = None
    try:
        # 4. Configure BenchMARL Task
        task = vmas_task.get_from_yaml(path=str(rl_cfg_dir / task_yaml_name))

        # 5. Configure Algorithm & Experiment
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

        path_parts = Path(output_dir).parts
        if "results" in path_parts:
            # Try to extract identifying parts from path structure:
            # .../results/DATE/EXPERIMENT_NAME/behavior_X/attempt_Y
            try:
                # Find index of 'results' and look ahead
                res_idx = path_parts.index("results")
                date_str = path_parts[res_idx + 1]  # e.g. 2026-01-19_05-27-13
                # behavior_X -> behavior_0 -> 0
                behavior_str = path_parts[res_idx + 3].split("_")[-1]
                # attempt_Y -> attempt_0 -> 0
                attempt_str = path_parts[res_idx + 4].split("_")[-1]

                run_name = f"{date_str}-B{behavior_str}-A{attempt_str}"
            except (IndexError, ValueError):
                # Fallback if path structure isn't as expected
                run_name = Path(output_dir).name
        else:
            run_name = Path(output_dir).name

        experiment_config.wandb_extra_kwargs = {"name": run_name, "tags": [scenario_name]}

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
        )

        # 6. Run Training
        experiment.run()

    finally:
        # 8. Restore original class to avoid side effects on other runs
        vmas.make_env = original_make_env

    return experiment


def rollout_vmas_env(policy, env, get_state_fn=None, **kwargs):
    policy.eval()

    # Use time and pid to ensure different seeds regardless of global torch seed
    seed = (int(time.time() * 1000) + os.getpid()) % 100000
    torch.manual_seed(seed)
    env.set_seed(seed)

    states = []
    images = []

    def render_callback(env, td):
        frame = env.render(mode="rgb_array")
        if frame is not None:
            images.append(frame)
        if get_state_fn is not None:
            states.append(get_state_fn(env))

    # Filter kwargs to only pass supported arguments to env.rollout
    rollout_sig = inspect.signature(env.rollout)
    supported_args = set(rollout_sig.parameters.keys())
    rollout_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}

    with torch.no_grad():
        rollout_td = env.rollout(
            policy=policy,
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
