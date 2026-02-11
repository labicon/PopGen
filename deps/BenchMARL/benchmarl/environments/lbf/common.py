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

from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, TensorSpec, OneHotDiscreteTensorSpec, DiscreteTensorSpec
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
    def __init__(self, env: gymnasium.Env, num_agents: int, device=DEVICE_TYPING):
        super().__init__(env, device=device)
        self.num_agents = num_agents
        
        # --- Fix Action Spec ---
        # Explicitly construct the spec to ensure consistency
        # Assuming all agents have same action dim (e.g. 6)
        # old_action_spec from GymWrapper might be OneHot(2, 6)
        
        # We need to know action_dim. 
        # For LBF 8x8-2p-2f, action_dim is 6.
        # We can inspect old_action_spec
        old_action_spec = self.action_spec
        action_dim = old_action_spec.shape[-1]
        
        # Use DiscreteTensorSpec(shape=(n_agents,)) instead of OneHot
        # This seems to play nicer with BenchMARL's expansion logic
        self.action_spec = CompositeSpec({
             "agents": CompositeSpec({
                 "action": DiscreteTensorSpec(
                     n=action_dim,
                     shape=torch.Size((self.num_agents,)),
                     device=self.device,
                     dtype=torch.long
                 )
             }, shape=(self.num_agents,))
        }, shape=self.batch_size, device=self.device)
        
        # --- Fix Observation Spec ---
        # GymWrapper creates a stacked spec for Tuple observation spaces
        old_obs_spec = self.observation_spec["observation"]
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": old_obs_spec
            }, shape=(self.num_agents,))
        }, shape=self.batch_size, device=self.device)
        
        # --- Fix Reward Spec ---
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec(
                    shape=torch.Size([self.num_agents, 1]), 
                    device=self.device
                )
            }, shape=(self.num_agents,))
        }, shape=self.batch_size, device=self.device)

    def _step(self, tensordict):
        # Read action
        action_tensor = tensordict.get(("agents", "action")) 
        
        # Standardize to numpy indices/values for LBF
        if tensordict.batch_size == torch.Size([]):
            if action_tensor.ndim > 1 and action_tensor.shape[-1] > 1:
                # Assuming OneHot -> Indices
                action_idx = action_tensor.argmax(dim=-1)
            else:
                action_idx = action_tensor
            
            # Convert to list for LBF
            action_np = action_idx.detach().cpu().numpy()
            action_list = tuple(action_np.tolist())
            
            # Step
            obs, reward, terminated, truncated, info = self.env.step(action_list)
            
            # Process output
            # obs: tuple of arrays -> (n_agents, obs_dim)
            obs_tensor = torch.tensor(np.stack(obs), device=self.device, dtype=torch.float32)
            
            # reward: list of floats -> (n_agents,)
            reward_tensor = torch.tensor(np.array(reward), device=self.device, dtype=torch.float32)
            # Add reward dim -> (n_agents, 1)
            reward_tensor = reward_tensor.unsqueeze(-1)
            
            done = terminated or truncated
            
            return TensorDict({
                "agents": TensorDict({
                    "observation": obs_tensor,
                    "reward": reward_tensor
                }, batch_size=(self.num_agents,), device=self.device),
                "done": torch.tensor(done, device=self.device),
                "terminated": torch.tensor(terminated, device=self.device),
                "truncated": torch.tensor(truncated, device=self.device),
            }, batch_size=torch.Size([]), device=self.device)
        else:
             raise NotImplementedError("Batch stepping not implemented yet for LbfGymWrapper")

    def render(self, *args, **kwargs):
        return self.env.render()

    def _reset(self, tensordict=None, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_tensor = torch.tensor(np.stack(obs), device=self.device, dtype=torch.float32)
        
        return TensorDict({
            "agents": TensorDict({
                "observation": obs_tensor
            }, batch_size=(self.num_agents,), device=self.device),
            "done": torch.tensor(False, device=self.device),
            "terminated": torch.tensor(False, device=self.device),
            "truncated": torch.tensor(False, device=self.device),
        }, batch_size=torch.Size([]), device=self.device)


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
        # The data in the tensordict will havebe presented this way
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