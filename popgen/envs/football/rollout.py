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

def get_state_football(env):
    state = {}

    blue_team, red_team = env.scenario.blue_agents, env.scenario.red_agents
    goal_pos = env.scenario.right_goal_pos

    # Ego team agents
    for i, agent in enumerate(blue_team):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        state[f"blue_agent{i}_pos"] = [round(x, 2) for x in pos]
        state[f"blue_agent{i}_vel"] = [round(x, 2) for x in vel]

    # Adversary team agents
    for i, agent in enumerate(red_team):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        state[f"red_agent{i}_pos"] = [round(x, 2) for x in pos]
        state[f"red_agent{i}_vel"] = [round(x, 2) for x in vel]

    # Goal position
    state["goal_pos"] = [round(x, 2) for x in goal_pos.cpu().numpy().tolist()]

    # Ball state
    ball_pos = env.scenario.ball.state.pos[0].cpu().numpy().tolist()
    ball_vel = env.scenario.ball.state.vel[0].cpu().numpy().tolist()
    state["ball_pos"] = [round(x, 2) for x in ball_pos]
    state["ball_vel"] = [round(x, 2) for x in ball_vel]

    return state

def rollout_behavior(agent0_policy: callable, agent1_policy: callable):
    # Create environment
    # We use 1 environment for manual control
    max_steps = 200
    env = make_env(
        scenario="football",
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        n_blue_agents=2,
        n_red_agents=2,
        ai_strength=0.5,
        ai_decision_strength=0.5,
        ai_precision_strength=0.5,
    )

    # Containers for trajectory and rollout images
    states = []
    images = []

    # Reset environment
    obs = env.reset()

    # Capture initial frame and state in rgb_array mode
    images.append(env.render(mode="rgb_array"))
    states.append(get_state_football(env))

    step = 0
    while step < max_steps:
        # Construct actions
        try:
            action_0 = agent0_policy(obs[0].squeeze(0))
            action_1 = agent1_policy(obs[1].squeeze(0))
        except Exception as e:
            return states, images, str(e)

        actions = [action_0, action_1]

        # Step environment
        obs, rews, dones, info = env.step(actions)

        # Record state and rendered image
        states.append(get_state_football(env))
        images.append(env.render(mode="rgb_array"))

        step += 1
        if dones.any():
            break

        # Control frame rate (optional)
        time.sleep(0.05)

    # Return trajectory of states and rollout images
    return states, images, None
