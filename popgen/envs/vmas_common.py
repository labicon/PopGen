import inspect
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import vmas
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from popgen.envs.util import apply_rollout_seed, compose_rollout_policy, parse_single_scripted_policy


def _build_vmas_train_group_map(
    task_config: dict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split the agents in ``task_config["group_map"]`` into ego and teammate groups.

    Two layouts are supported:

    **Football layout** – ``group_map`` contains both ``"agent_blue"`` (our
    team) and ``"agent_red"`` (opponent).  ``"agent_red"`` is left untouched;
    ``"agent_blue"`` is split so the first blue agent becomes ``"ego"`` and
    the remaining blue agents become ``"teammate"``.

    **General layout** – all agents are flattened across groups in insertion
    order; the first agent becomes ``"ego"`` and the rest become
    ``"teammate"``.

    Args:
        task_config: BenchMARL task configuration dict.  Must contain a
            non-empty ``"group_map"`` with at least two agents in the
            trainable group.

    Returns:
        A ``(group_map, trainable_group)`` tuple where:

        - *group_map* – new group map with ``"ego"``, ``"teammate"``, and
          (football only) ``"agent_red"`` entries.
        - *trainable_group* – ``{"ego": [first_agent]}``, the subset whose
          parameters are updated during training.

    Raises:
        ValueError: If ``"group_map"`` is absent/empty or the trainable team
            contains fewer than two agents.
    """
    original_group_map = task_config.get("group_map")

    # Football layout: "agent_blue" (our team) vs "agent_red" (opponent, unchanged).
    if original_group_map and "agent_blue" in original_group_map and "agent_red" in original_group_map:
        blue_agents = list(original_group_map["agent_blue"])
        if len(blue_agents) < 2:
            raise ValueError(
                f"Need at least 2 blue agents to form ego/teammate split; got {blue_agents}."
            )
        group_map = {
            "ego": [blue_agents[0]],
            "teammate": blue_agents[1:],
            "agent_red": list(original_group_map["agent_red"]),
        }
        trainable_group = {"ego": [blue_agents[0]]}
        return group_map, trainable_group

    # General layout:
    # 1) if group_map exists, flatten all agents across groups, first → ego, rest → teammate.
    # 2) otherwise infer agent names from count fields.
    if original_group_map:
        seen: set[str] = set()
        agent_names: list[str] = []
        for grouped_agents in original_group_map.values():
            for name in grouped_agents:
                if name not in seen:
                    seen.add(name)
                    agent_names.append(name)
    else:
        n_agents = task_config.get("n_agents")
        if n_agents is not None:
            agent_names = [f"agent_{i}" for i in range(n_agents)]
        elif n_agents is None and {"n_blue_agents", "n_red_agents"} <= set(task_config.keys()):
            n_agents = task_config["n_blue_agents"]
            agent_names = [f"agent_blue_{i}" for i in range(n_agents)]
        else:
            raise ValueError("Could not infer number of agents to build VMAS group_map.")

    if len(agent_names) < 2:
        raise ValueError(
            f"Need at least 2 agents to form ego/teammate split; got {agent_names}."
        )

    ego_group = [agent_names[0]]
    teammate_group = agent_names[1:]

    group_map = {"ego": ego_group, "teammate": teammate_group}
    trainable_group = {"ego": ego_group}
    return group_map, trainable_group


def train_vmas_env(
    reward_fn,
    output_dir,
    scenario_module,
    scenario_name,
    vmas_task,
    rl_cfg_dir,
    teammate_policy: list[nn.Module | Callable] | None = None,
    task_yaml_name=None,
    ippo_yaml_name="ippo.yaml",
    experiment_yaml_name="base_experiment.yaml",
    mlp_yaml_name="mlp.yaml",
    callbacks=None,
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

        group_map, trainable_group = _build_vmas_train_group_map(task_config=task.config)
        task.config["group_map"] = group_map

        callbacks = list(callbacks or [])

        if teammate_policy:
            from popgen.agents.scripted_partner import TeammatePolicyCallback

            callbacks.append(
                TeammatePolicyCallback(
                    teammate_policies=teammate_policy,
                    trainable_group=trainable_group,
                )
            )

        # 5. Configure Algorithm & Experiment
        ippo_path = rl_cfg_dir / ippo_yaml_name
        if ippo_path.exists():
            algorithm_config = IppoConfig.get_from_yaml(path=str(ippo_path))
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
            callbacks=callbacks,
        )

        # 6. Run Training
        experiment.run()

    finally:
        # 8. Restore original class to avoid side effects on other runs
        vmas.make_env = original_make_env

    return experiment


def rollout_vmas_env(policy, env, get_state_fn=None, **kwargs):
    """Rollout a policy input that may be a single policy or a list of policies."""
    policy = compose_rollout_policy(policy)
    policy.eval()

    apply_rollout_seed(env)

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
            trust_policy=True,
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
