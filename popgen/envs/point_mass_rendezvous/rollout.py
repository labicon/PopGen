import time
import torch
import os
import sys
import shutil
import numpy as np
from PIL import Image
from vmas import make_env

# def agent0_policy(obs):
#     return torch.tensor([[0.0, 0.0]])

# def agent1_policy(obs):
#     return torch.tensor([[0.0, 0.0]])

def get_state_point_mass_rendezvous(env):
    state = {}
    for i, agent in enumerate(env.world.agents):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        state[f"agent{i}_pos"] = [round(x, 2) for x in pos]
        state[f"agent{i}_vel"] = [round(x, 2) for x in vel]

    for i, landmark in enumerate(env.world.landmarks):
        pos = landmark.state.pos[0].cpu().numpy().tolist()
        state[f"landmark{i}_pos"] = [round(x, 2) for x in pos]

    return state

def rollout_behavior(agent0_policy: callable, agent1_policy: callable):
    # Create environment
    # We use 1 environment for manual control
    max_steps = 200
    env = make_env(
        scenario="point_mass_rendezvous",
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        n_agents=2,
        mode="PMR-C",
    )

    # Containers for trajectory and rollout images
    states = []
    images = []

    # Reset environment
    obs = env.reset()

    # Capture initial frame and state in rgb_array mode
    images.append(env.render(mode="rgb_array"))
    states.append(get_state_point_mass_rendezvous(env))

    step = 0
    while step < max_steps:
        # Construct actions
        try:
            action_0 = agent0_policy(obs[0].squeeze(0))
            action_1 = agent1_policy(obs[1].squeeze(0))
        except Exception as e:
            return states, images, str(e)
        
        # Step environment
        obs, rews, dones, info = env.step([action_0, action_1])

        # Record state and rendered image
        states.append(get_state_point_mass_rendezvous(env))
        images.append(env.render(mode="rgb_array"))

        step += 1
        if dones.any():
            break

        # Control frame rate (optional)
        # time.sleep(0.05)

    # Return trajectory of states and rollout images
    return states, images, None