#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_landmarks: int = MISSING
    mode: str = MISSING
    shared_goal_shaping: bool = False
    shared_goal_shaping_scale: float = 1.0
    shared_goal_shaping_gamma: float = 0.99
    shared_goal_shaping_temperature: float = 0.25
    shared_goal_shaping_pair_weight: float = 0.05
