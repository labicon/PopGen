#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

import gymnasium
import lbforaging
import numpy as np
import torch
from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDict, TensorDictBase

from torchrl.data import Categorical, CompositeSpec, UnboundedContinuousTensorSpec, TensorSpec, OneHotDiscreteTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.libs import GymWrapper

class LbfTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/lbf

    LBF = None

    @staticmethod
    def associated_class():
        return LbfClass


class LbfGymWrapper(GymWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        num_agents: int,
        device=DEVICE_TYPING,
        group_map: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(env, device=device)
        self.num_agents = num_agents
        self.group_map = group_map

        old_action_spec = self.action_spec
        action_dim = old_action_spec.shape[-1]

        old_obs_spec = self.observation_spec["observation"]

        if self.group_map:
            obs_dim = old_obs_spec.shape[-1]
            obs_spec_dict = {}
            action_spec_dict = {}
            reward_spec_dict = {}

            for gname, names in self.group_map.items():
                n = len(names)
                obs_spec_dict[gname] = CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            shape=torch.Size([n, obs_dim]),
                            device=self.device,
                            dtype=torch.float32,
                        )
                    },
                    shape=(n,),
                )
                action_spec_dict[gname] = CompositeSpec(
                    {
                        "action": Categorical(
                            n=action_dim,
                            shape=torch.Size((n,)),
                            device=self.device,
                            dtype=torch.long,
                        )
                    },
                    shape=(n,),
                )
                reward_spec_dict[gname] = CompositeSpec(
                    {
                        "reward": UnboundedContinuousTensorSpec(
                            shape=torch.Size([n, 1]),
                            device=self.device,
                        )
                    },
                    shape=(n,),
                )

            self.observation_spec = CompositeSpec(
                obs_spec_dict, shape=self.batch_size, device=self.device
            )
            self.action_spec = CompositeSpec(
                action_spec_dict, shape=self.batch_size, device=self.device
            )
            self.reward_spec = CompositeSpec(
                reward_spec_dict, shape=self.batch_size, device=self.device
            )
        else:
            self.action_spec = CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "action": Categorical(
                                n=action_dim,
                                shape=torch.Size((self.num_agents,)),
                                device=self.device,
                                dtype=torch.long,
                            )
                        },
                        shape=(self.num_agents,),
                    )
                },
                shape=self.batch_size,
                device=self.device,
            )

            self.observation_spec = CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {"observation": old_obs_spec}, shape=(self.num_agents,)
                    )
                },
                shape=self.batch_size,
                device=self.device,
            )

            self.reward_spec = CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "reward": UnboundedContinuousTensorSpec(
                                shape=torch.Size([self.num_agents, 1]),
                                device=self.device,
                            )
                        },
                        shape=(self.num_agents,),
                    )
                },
                shape=self.batch_size,
                device=self.device,
            )

    def _step(self, tensordict):
        if tensordict.batch_size != torch.Size([]):
            raise NotImplementedError("Batch stepping not implemented yet for LbfGymWrapper")

        if self.group_map:
            # Merge per-group actions into a global action list ordered by agent index
            action_list = [None] * self.num_agents
            for gname, names in self.group_map.items():
                g_action = tensordict.get((gname, "action"))
                for local_i, aname in enumerate(names):
                    global_idx = int(aname.split("_")[1])
                    action_list[global_idx] = g_action[local_i].item()

            obs, reward, terminated, truncated, info = self.env.step(tuple(action_list))

            obs_tensor = torch.tensor(np.stack(obs), device=self.device, dtype=torch.float32)
            reward_tensor = torch.tensor(
                np.array(reward), device=self.device, dtype=torch.float32
            )
            done = terminated or truncated

            out_td = TensorDict(
                {
                    "done": torch.tensor(done, device=self.device),
                    "terminated": torch.tensor(terminated, device=self.device),
                    "truncated": torch.tensor(truncated, device=self.device),
                },
                batch_size=torch.Size([]),
                device=self.device,
            )
            for gname, names in self.group_map.items():
                idxs = [int(n.split("_")[1]) for n in names]
                n = len(names)
                out_td[gname] = TensorDict(
                    {
                        "observation": obs_tensor[idxs],
                        "reward": reward_tensor[idxs].unsqueeze(-1),
                    },
                    batch_size=(n,),
                    device=self.device,
                )
            return out_td
        else:
            # Read action
            action_tensor = tensordict.get(("agents", "action"))

            if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
                action_idx = action_tensor.argmax(dim=-1)
            elif action_tensor.ndim > 1 and action_tensor.shape[-1] == 1:
                action_idx = action_tensor.squeeze(-1)
            else:
                action_idx = action_tensor

            action_np = action_idx.detach().cpu().numpy()
            action_list = tuple(action_np.tolist())

            obs, reward, terminated, truncated, info = self.env.step(action_list)

            obs_tensor = torch.tensor(np.stack(obs), device=self.device, dtype=torch.float32)
            reward_tensor = torch.tensor(
                np.array(reward), device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            done = terminated or truncated

            return TensorDict(
                {
                    "agents": TensorDict(
                        {"observation": obs_tensor, "reward": reward_tensor},
                        batch_size=(self.num_agents,),
                        device=self.device,
                    ),
                    "done": torch.tensor(done, device=self.device),
                    "terminated": torch.tensor(terminated, device=self.device),
                    "truncated": torch.tensor(truncated, device=self.device),
                },
                batch_size=torch.Size([]),
                device=self.device,
            )

    def render(self, *args, **kwargs):
        return self.env.render()

    def _reset(self, tensordict=None, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_tensor = torch.tensor(np.stack(obs), device=self.device, dtype=torch.float32)

        if self.group_map:
            out_td = TensorDict(
                {
                    "done": torch.tensor(False, device=self.device),
                    "terminated": torch.tensor(False, device=self.device),
                    "truncated": torch.tensor(False, device=self.device),
                },
                batch_size=torch.Size([]),
                device=self.device,
            )
            for gname, names in self.group_map.items():
                idxs = [int(n.split("_")[1]) for n in names]
                n = len(names)
                out_td[gname] = TensorDict(
                    {"observation": obs_tensor[idxs]},
                    batch_size=(n,),
                    device=self.device,
                )
            return out_td
        else:
            return TensorDict(
                {
                    "agents": TensorDict(
                        {"observation": obs_tensor},
                        batch_size=(self.num_agents,),
                        device=self.device,
                    ),
                    "done": torch.tensor(False, device=self.device),
                    "terminated": torch.tensor(False, device=self.device),
                    "truncated": torch.tensor(False, device=self.device),
                },
                batch_size=torch.Size([]),
                device=self.device,
            )


class LbfClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        def make_env():
            env_name = self.config["env_name"]
            num_agents = self.config["players"]
            gym_env = gymnasium.make(env_name, render_mode="rgb_array")

            env = LbfGymWrapper(
                env=gym_env,
                num_agents=num_agents,
                device=device,
                group_map=self.config.get("group_map"),
            )

            return env

        return make_env


    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return False

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return True

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return True

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return self.config.get("max_episode_steps", 50)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will be presented this way
        if getattr(env, "group_map", None):
            return env.group_map
        num_agents = self.config["players"]
        return {"agents": [f"agent_{i}" for i in range(num_agents)]}

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the observation.
        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        return env.observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the action.
        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        return env.action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        return None

    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "lbf"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}