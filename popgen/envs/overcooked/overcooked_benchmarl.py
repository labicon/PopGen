from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import yaml
from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDict, TensorDictBase

from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

NUM_ACTIONS = len(Action.INDEX_TO_ACTION)  # 6: N, S, E, W, STAY, INTERACT
RL_CFG_PATH = Path(__file__).parent / "rl_cfg" / "overcooked.yaml"
LARGE_ROOM_COMPACT_OBS_INDICES = (
    list(range(0, 10))
    + list(range(12, 22))
    + list(range(22, 32))
    + list(range(42, 46))
    + list(range(92, 96))
)


@lru_cache(maxsize=1)
def _load_default_task_config() -> dict:
    with RL_CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_default_layout_name() -> str:
    return _load_default_task_config().get("layout_name", "cramped_room")


def _observation_indices_for_layout(layout_name: str) -> list[int] | None:
    if layout_name == "large_room":
        return list(LARGE_ROOM_COMPACT_OBS_INDICES)
    return None


class _MdpGenerator:
    """Picklable MDP factory for OvercookedEnv (avoids lambda which can't be pickled)."""

    def __init__(self, layout_name: str):
        self.layout_name = layout_name

    def __call__(self, _ignored=None):
        return OvercookedGridworld.from_layout_name(self.layout_name)


class OvercookedBenchWrapper(EnvBase):
    """TorchRL EnvBase wrapper around OvercookedEnv for BenchMARL.

    Groups agents as {"ego": ["agent_0"], "teammate": ["agent_1"]} by default,
    matching the popgen ego/teammate split convention.
    """

    def __init__(
        self,
        layout_name: Optional[str] = None,
        horizon: int = 400,
        device: DEVICE_TYPING = "cpu",
        group_map: Optional[Dict[str, List[str]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(device=device, batch_size=[])

        layout_name = layout_name or _get_default_layout_name()
        self.layout_name = layout_name
        self.horizon = horizon
        self.group_map = group_map or {"ego": ["agent_0"], "teammate": ["agent_1"]}
        self._obs_indices = _observation_indices_for_layout(self.layout_name)

        # Build OvercookedEnv using a picklable generator (lambda would break BenchMARL's pickle)
        self.oc_env = OvercookedEnv(
            mdp_generator_fn=_MdpGenerator(layout_name),
            horizon=horizon,
            info_level=0,
        )

        # Determine obs_dim from the initial state (reset is called in OvercookedEnv.__init__)
        feats = self._project_features(self.oc_env.featurize_state_mdp(self.oc_env.state))
        obs_dim = len(feats[0])
        self._obs_dim = obs_dim

        # Build per-group specs
        obs_spec_dict: Dict = {}
        action_spec_dict: Dict = {}
        reward_spec_dict: Dict = {}

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
                    "action": DiscreteTensorSpec(
                        n=NUM_ACTIONS,
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
        self.done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
                ),
                "terminated": DiscreteTensorSpec(
                    n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
                ),
                "truncated": DiscreteTensorSpec(
                    n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
                ),
            },
            shape=self.batch_size,
            device=self.device,
        )

        if seed is not None:
            self.set_seed(seed)

    def _project_features(self, feats):
        if self._obs_indices is None:
            return feats
        return [feat[self._obs_indices] for feat in feats]

    def _reset(self, tensordict=None, **kwargs):
        self.oc_env.reset()
        feats = self._project_features(self.oc_env.featurize_state_mdp(self.oc_env.state))

        out_td = TensorDict(
            {
                "done": torch.tensor([False], device=self.device),
                "terminated": torch.tensor([False], device=self.device),
                "truncated": torch.tensor([False], device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

        agent_feats = [
            torch.tensor(f, device=self.device, dtype=torch.float32) for f in feats
        ]
        feat_idx = 0
        for gname, names in self.group_map.items():
            n = len(names)
            obs = torch.stack(agent_feats[feat_idx : feat_idx + n], dim=0)
            out_td[gname] = TensorDict(
                {"observation": obs},
                batch_size=(n,),
                device=self.device,
            )
            feat_idx += n

        return out_td

    def _step(self, tensordict):
        # Map group actions to joint action list
        action_list: List = [None, None]
        for gname, names in self.group_map.items():
            g_action = tensordict.get((gname, "action"))
            for local_i, aname in enumerate(names):
                global_idx = int(aname.split("_")[1])
                action_list[global_idx] = Action.INDEX_TO_ACTION[
                    int(g_action[local_i].item())
                ]

        next_state, sparse_reward, done, info = self.oc_env.step(tuple(action_list))

        feats = self._project_features(self.oc_env.featurize_state_mdp(next_state))
        agent_feats = [
            torch.tensor(f, device=self.device, dtype=torch.float32) for f in feats
        ]
        reward_val = float(sparse_reward)

        out_td = TensorDict(
            {
                "done": torch.tensor([done], device=self.device),
                "terminated": torch.tensor([done], device=self.device),
                "truncated": torch.tensor([False], device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

        feat_idx = 0
        for gname, names in self.group_map.items():
            n = len(names)
            obs = torch.stack(agent_feats[feat_idx : feat_idx + n], dim=0)
            rewards = torch.tensor(
                [reward_val] * n, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            out_td[gname] = TensorDict(
                {"observation": obs, "reward": rewards},
                batch_size=(n,),
                device=self.device,
            )
            feat_idx += n

        return out_td

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def render(self, mode="rgb_array"):
        try:
            import pygame
            from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

            vis = StateVisualizer()
            surface = vis.render_state(
                self.oc_env.state, grid=self.oc_env.mdp.terrain_mtx
            )
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        except Exception:
            return None


class OvercookedTask(Task):
    OVERCOOKED = None

    @staticmethod
    def associated_class():
        return OvercookedClass


class OvercookedClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        def make_env():
            layout_name = self.config.get("layout_name", _get_default_layout_name())
            horizon = self.config.get("horizon", 400)
            group_map = self.config.get("group_map")
            return OvercookedBenchWrapper(
                layout_name=layout_name,
                horizon=horizon,
                device=device,
                group_map=group_map,
                seed=seed,
            )

        return make_env

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("horizon", 400)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if getattr(env, "group_map", None):
            return env.group_map
        return {"ego": ["agent_0"], "teammate": ["agent_1"]}

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        return env.observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    @staticmethod
    def env_name() -> str:
        return "overcooked"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        return {}
