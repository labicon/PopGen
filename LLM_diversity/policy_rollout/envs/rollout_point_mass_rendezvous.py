import time
import torch
import os
import shutil
import numpy as np
from PIL import Image
from vmas import make_env
from policy_rollout.envs.agent1_policy import agent1_policy

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
            action_0 = agent1_policy(obs[0].squeeze(0))
            action_1 = agent2_policy(obs[1].squeeze(0))
        except Exception:
            return None, None

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
    return states, images

# def agent1_policy(obs):
#     return torch.tensor([[0.0, 0.0]])

def agent2_policy(obs):
    return torch.tensor([[0.0, 0.0]])

def get_state_point_mass_rendezvous(env):
    state = {}
    for i, agent in enumerate(env.world.agents):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        state[f"agent{i+1}_pos"] = [round(x, 2) for x in pos]
        state[f"agent{i+1}_vel"] = [round(x, 2) for x in vel]

    for i, landmark in enumerate(env.world.landmarks):
        pos = landmark.state.pos[0].cpu().numpy().tolist()
        state[f"landmark{i+1}_pos"] = [round(x, 2) for x in pos]

    return state

def save_rollout(states, images, folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

    # Save 5 evenly distributed images
    num_images = 5
    if len(images) > 0:
        img_indices = np.linspace(0, len(images) - 1, num_images, dtype=int)
        # Use unique indices to avoid duplicates if len(images) < num_images
        img_indices = np.unique(img_indices)
        
        for idx in img_indices:
            img_data = images[idx]
            # Handle if render returns a list (common in vectorized envs)
            if isinstance(img_data, list):
                img_array = img_data[0]
            else:
                img_array = img_data
                
            img = Image.fromarray(img_array.astype(np.uint8))
            # Save with step index in filename for clarity
            img.save(os.path.join(folder_path, f"image_{idx}.png"))

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

def rollout_behavior(save_path: str = "results/rollout/test"):
    states, images = test_behavior_following()
    save_rollout(states, images, save_path)

if __name__ == "__main__":
    states, images = test_behavior_following()
    save_rollout(states, images, "results/rollout")