import inspect
import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import vmas
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from popgen.envs.utils import apply_rollout_seed


def default_trainable_group(group_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Mark only the semantic ``ego`` group as trainable."""
    ego_group = list(group_map.get("ego", []))
    if not ego_group:
        raise ValueError(f"Expected semantic group_map with non-empty 'ego', got {group_map}.")
    return {"ego": ego_group}


def train_vmas_env(
    reward_fn,
    output_dir,
    scenario_module,
    scenario_name,
    vmas_task,
    rl_cfg_dir,
    group_map: dict[str, list[str]] | None = None,
    trainable_group: dict[str, list[str]] | None = None,
    teammate_policy: list[nn.Module | Callable] | None = None,
    scenario_kwargs: dict | None = None,
    task_yaml_name=None,
    mappo_yaml_name="mappo.yaml",
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
                # Let the original scenario compute any internal shared-reward
                # caches (e.g., simple_tag's `adverary_rew`) used by non-ego agents.
                _ = super().reward(agent)
                batch_dim = self.world.batch_dim
                self.rew = reward_fn(self, agent)
                if self.rew.shape != (batch_dim,):
                    self.rew = self.rew.squeeze()
                    if self.rew.ndim == 0:
                        self.rew = self.rew.view(1)
                # Keep per-agent reward fields coherent with the overridden ego reward.
                agent.rew = self.rew
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
        if scenario_kwargs:
            task.config.update(scenario_kwargs)

        if group_map is None:
            configured_group_map = task.config.get("group_map")
            if not configured_group_map:
                raise ValueError(
                    "train_vmas_env requires an explicit semantic group_map or task.config['group_map']."
                )
            group_map = {name: list(agent_names) for name, agent_names in configured_group_map.items()}
        if trainable_group is None:
            trainable_group = default_trainable_group(group_map)

        task.config["group_map"] = group_map

        callbacks = list(callbacks or [])

        if teammate_policy:
            from popgen.agents.callback import TeammatePolicyCallback

            callbacks.append(
                TeammatePolicyCallback(
                    teammate_policies=teammate_policy,
                    trainable_group=trainable_group,
                )
            )

        # 5. Configure Algorithm & Experiment
        mappo_path = rl_cfg_dir / mappo_yaml_name
        ippo_path = rl_cfg_dir / ippo_yaml_name
        if ippo_path.exists():
            algorithm_config = IppoConfig.get_from_yaml(path=str(ippo_path))
        else:
            algorithm_config = MappoConfig.get_from_yaml(path=str(mappo_path))

        experiment_config = ExperimentConfig.get_from_yaml(path=str(rl_cfg_dir / experiment_yaml_name))
        model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / mlp_yaml_name))
        critic_model_config = MlpConfig.get_from_yaml(path=str(rl_cfg_dir / mlp_yaml_name))

        # Disable BenchMARL internal evaluation; B2T evaluates policies via
        # pipeline rollouts/adherence.
        experiment_config.evaluation = False
        experiment_config.evaluation_episodes = 1

        # BenchMARL defaults to CUDA in our configs; fall back to CPU if CUDA is unavailable.
        requested_devices = [
            str(getattr(experiment_config, "sampling_device", "")),
            str(getattr(experiment_config, "train_device", "")),
            str(getattr(experiment_config, "buffer_device", "")),
        ]
        if any(dev.startswith("cuda") for dev in requested_devices) and not torch.cuda.is_available():
            logging.warning(
                "CUDA devices were requested in experiment config but CUDA is unavailable. "
                "Falling back to CPU devices for training."
            )
            experiment_config.sampling_device = "cpu"
            experiment_config.train_device = "cpu"
            if str(getattr(experiment_config, "buffer_device", "")).lower() != "disk":
                experiment_config.buffer_device = "cpu"

        # B2T does not consume BenchMARL logger outputs; disable them to avoid
        # wandb/console issues interrupting training.
        experiment_config.loggers = []

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


def _collect_step_state(td) -> dict:
    """Extract canonical rollout teams:
    {"ego_team": {...}, "opponent_team": {...}}.
    """
    nested_keys = set(td.keys(include_nested=True))
    groups = [
        key
        for key in td.keys()
        if isinstance(key, str) and (key, "observation") in nested_keys
    ]

    result = {"ego_team": {}, "opponent_team": {}}
    ego_assigned = False
    teammate_idx = 0
    opponent_idx = 0

    for group in groups:
        obs = td[group, "observation"]

        try:
            act = td[group, "action"]
        except KeyError:
            act = None

        try:
            rew = td["next", group, "reward"]
        except KeyError:
            try:
                rew = td[group, "reward"]
            except KeyError:
                rew = None

        obs_2d = obs.reshape(-1, obs.shape[-1])
        num_agents = obs_2d.shape[0]
        act_2d = act.reshape(num_agents, -1) if act is not None else None
        rew_2d = rew.reshape(num_agents, -1) if rew is not None else None

        team = "ego" if group in ("ego", "teammate") else "opponent"
        for i in range(num_agents):
            action_val = None
            if act_2d is not None:
                action_item = act_2d[i].cpu().tolist()
                action_val = action_item[0] if len(action_item) == 1 else action_item

            reward_val = None
            if rew_2d is not None:
                reward_item = rew_2d[i].cpu().tolist()
                reward_val = reward_item[0] if len(reward_item) == 1 else reward_item

            payload = {
                "state": obs_2d[i].cpu().tolist(),
                "action": action_val,
                "reward": reward_val,
            }

            if team == "ego":
                if not ego_assigned:
                    canonical_name = "ego_agent"
                    ego_assigned = True
                else:
                    canonical_name = f"teammate_{teammate_idx}"
                    teammate_idx += 1
                result["ego_team"][canonical_name] = payload
            else:
                canonical_name = f"opponent_{opponent_idx}"
                opponent_idx += 1
                result["opponent_team"][canonical_name] = payload

    return result


def _extract_step_group_rewards(
    rollout_td,
    group: str,
    step_idx: int,
    num_steps: int,
) -> list[float]:
    key = ("next", group, "reward")
    if key not in rollout_td.keys(include_nested=True):
        return []

    rew_t = rollout_td[key]
    if rew_t.ndim == 0:
        return [float(rew_t.item())]

    if rew_t.ndim >= 2 and int(rew_t.shape[0]) == int(num_steps):
        rew_step = rew_t[step_idx]
    elif rew_t.ndim >= 2 and int(rew_t.shape[1]) == int(num_steps):
        rew_step = rew_t[:, step_idx]
    elif rew_t.shape[0] > step_idx:
        rew_step = rew_t[step_idx]
    else:
        rew_step = rew_t

    return [float(x) for x in rew_step.reshape(-1).detach().cpu().tolist()]


def _attach_rewards_from_rollout(states: list[dict[str, dict]], rollout_td) -> None:
    if not states:
        return

    nested_keys = set(rollout_td.keys(include_nested=True))
    top_groups = [k for k in rollout_td.keys() if isinstance(k, str)]
    reward_groups = [g for g in top_groups if ("next", g, "reward") in nested_keys]
    num_steps = len(states)

    for step_idx, step in enumerate(states):
        ego_names = []
        if "ego_agent" in step.get("ego_team", {}):
            ego_names.append("ego_agent")
        ego_names.extend(
            sorted(
                [k for k in step.get("ego_team", {}) if k.startswith("teammate_")],
                key=lambda x: int(x.split("_")[1]),
            )
        )
        opp_names = sorted(
            [k for k in step.get("opponent_team", {}) if k.startswith("opponent_")],
            key=lambda x: int(x.split("_")[1]),
        )

        ego_vals: list[float] = []
        opp_vals: list[float] = []
        for g in reward_groups:
            vals = _extract_step_group_rewards(rollout_td=rollout_td, group=g, step_idx=step_idx, num_steps=num_steps)
            team = "ego" if g in ("ego", "teammate") else "opponent"
            if team == "ego":
                ego_vals.extend(vals)
            else:
                opp_vals.extend(vals)

        for name, val in zip(ego_names, ego_vals):
            step["ego_team"][name]["reward"] = float(val)
        for name, val in zip(opp_names, opp_vals):
            step["opponent_team"][name]["reward"] = float(val)


def rollout_vmas_env(policy, env, seed: int | None = None, **kwargs):
    """Rollout a single policy module and collect per-step state from rollout tensordicts."""
    if not hasattr(policy, "eval"):
        raise TypeError(
            f"rollout_vmas_env expects a single policy module with eval(), got {type(policy).__name__}."
        )
    policy.eval()

    apply_rollout_seed(env, seed=seed)

    if "get_state_fn" in kwargs:
        raise TypeError(
            "'get_state_fn' is no longer supported in rollout_vmas_env; "
            "rollout states are now collected directly from rollout td."
        )

    states = []
    images = []

    def render_callback(env, td):
        try:
            frame = env.render(mode="rgb_array")
        except Exception:
            frame = None
        if frame is not None:
            images.append(frame)
        states.append(_collect_step_state(td))

    # Filter kwargs to only pass supported arguments to env.rollout
    rollout_sig = inspect.signature(env.rollout)
    supported_args = set(rollout_sig.parameters.keys())
    rollout_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}

    with torch.no_grad():
        rollout_td = env.rollout(
            policy=policy,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_all_done=True,
            trust_policy=True,
            **rollout_kwargs,
        )
    _attach_rewards_from_rollout(states=states, rollout_td=rollout_td)

    return states, images, None
