import time
import torch
import os
import sys
import shutil
import numpy as np
from PIL import Image
from vmas import make_env

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from policy_rollout.envs.agent0_policy import agent0_policy, agent1_policy

def test_behavior_following(max_steps: int = 200):
    # Create environment
    # We use 1 environment for manual control
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

        actions = [action_0, action_1]

        # Step environment
        obs, rews, dones, info = env.step(actions)

        # Record state and rendered image
        states.append(get_state_point_mass_rendezvous(env))
        images.append(env.render(mode="rgb_array"))

        step += 1
        if dones.any():
            break

        # Control frame rate (optional)
        time.sleep(0.05)

    # Return trajectory of states and rollout images
    return states, images, None

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

def save_rollout(states, images, folder_path, attempt: int = 0):
    # if os.path.exists(folder_path):
    #     shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

    # Save images as GIF
    if len(images) > 0:
        frames = []
        for img_data in images:
            # Handle if render returns a list (common in vectorized envs)
            if isinstance(img_data, list):
                img_array = img_data[0]
            else:
                img_array = img_data
            
            frames.append(Image.fromarray(img_array.astype(np.uint8)))
            
        if frames:
            frames[0].save(
                os.path.join(folder_path, "rollout.gif"),
                save_all=True,
                append_images=frames[1:],
                duration=50,  # Duration between frames in milliseconds
                loop=0
            )

            # Also save as rollout_{attempt}.gif
            if attempt >= 0:
                frames[0].save(
                    os.path.join(folder_path, f"rollout_{attempt}.gif"),
                    save_all=True,
                    append_images=frames[1:],
                    duration=50,
                    loop=0
                )

    # Save 200 evenly distributed states
    num_states = 20
    if len(states) <= num_states:
        state_indices = np.arange(len(states))
    else:
        state_indices = np.linspace(0, len(states) - 1, num_states, dtype=int)
        state_indices = np.unique(state_indices)

    with open(os.path.join(folder_path, "trajectory.txt"), "w") as f:
        for idx in state_indices:
            state = states[idx]
            f.write(f"Step {idx}: {state}\n")

def rollout_behavior(save_path: str = "results/rollout/test", attempt: int = 0):
    states, images, error = test_behavior_following()
    if error:
        return error
    save_rollout(states, images, save_path, attempt)
    return None

if __name__ == "__main__":
    states, images, error = test_behavior_following()
    if error:
        print(f"Rollout failed: {error}")
    else:
        save_rollout(states, images, "results/test_rollout")