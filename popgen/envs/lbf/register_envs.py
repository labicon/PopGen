"""Custom LBF environment registrations.

Import this module (or ensure it is imported) before calling gymnasium.make()
with any of the IDs defined here.
"""

import gymnasium as gym

# Variant of Foraging-8x8-2p-2f-coop-v3 where the two food items always have
# distinct levels, enforcing differentiated coordination demands.
# food 0: level 2  (requires combined player level >= 2)
# food 1: level 3  (requires combined player level >= 3, i.e. both players)
gym.register(
    id="Foraging-8x8-2p-2f-coop-difflevels-v3",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "min_player_level": 2,
        "max_player_level": 2,
        "field_size": (8, 8),
        "min_food_level": [3, 4],
        "max_food_level": [3, 4],
        "max_num_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
        "penalty": 0.0,
        "collision_penalty": 0.0,
    },
)
