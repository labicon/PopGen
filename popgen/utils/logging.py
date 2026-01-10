import json
import os
import numpy as np
from PIL import Image

def _save_rollout_gif(images, save_path, filename="rollout.gif"):
    # Save images as GIF
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
            os.path.join(save_path, filename),
            save_all=True,
            append_images=frames[1:],
            duration=50,  # Duration between frames in milliseconds
            loop=0
        )

def _save_rollout_data(rollout_data, save_path, num_states: int = 20):
    # Sample states to save
    if len(rollout_data) <= num_states:
        state_indices = np.arange(len(rollout_data))
    else:
        state_indices = np.linspace(0, len(rollout_data) - 1, num_states, dtype=int)
        state_indices = np.unique(state_indices)

    # Save rollout states as jsonl
    with open(os.path.join(save_path, "rollout_data.jsonl"), "w") as f:
        for idx in state_indices:
            state = rollout_data[idx]
            state["step"] = idx
            # ensure all numpy types are converted to native python types
            for key, value in state.items():
                if isinstance(value, np.generic):
                    state[key] = value.item()
            f.write(f"{json.dumps(state)}\n")

# TODO: check this
def save_rollout(rollout_data, images, save_path, num_states: int = 20):
    # Ensure the folder exists
    os.makedirs(save_path, exist_ok=True)

    # Save images as GIF
    if len(images) > 0:
        _save_rollout_gif(images, save_path, filename=f"rollout.gif")

    # Save rollout data
    _save_rollout_data(rollout_data, save_path, num_states=num_states)