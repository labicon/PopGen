import json
import os
from typing import Any

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
            loop=0,
        )


def _save_rollout_data(rollout_data, save_path):
    # Save rollout states as jsonl
    with open(os.path.join(save_path, "rollout_data.jsonl"), "w") as f:
        for idx in range(len(rollout_data)):
            state = rollout_data[idx]
            state["step"] = idx
            # ensure all numpy types are converted to native python types
            for key, value in state.items():
                if isinstance(value, np.generic):
                    state[key] = value.item()
            f.write(f"{json.dumps(state)}\n")


# TODO: check this
def save_rollout(rollout_data, images, save_path):
    # Ensure the folder exists
    os.makedirs(save_path, exist_ok=True)

    # Save images as GIF
    if len(images) > 0:
        _save_rollout_gif(images, save_path, filename="rollout.gif")

    # Save rollout data
    _save_rollout_data(rollout_data, save_path)


def uniform_sample(items: list[Any], n_samples: int) -> list[Any]:
    """
    Uniformly sample items from a list.

    Args:
        items: List of items to sample from.
        n_samples: Maximum number of items to return.

    Returns:
        List of uniformly sampled items.
    """
    if len(items) <= n_samples:
        return items
    indices = np.linspace(0, len(items) - 1, n_samples, dtype=int)
    indices = np.unique(indices)
    return [items[i] for i in indices]


def load_rollout(rollout_path: str, states_samples: int = 20, images_samples: int = 5) -> tuple[list[dict[str, Any]] | None, list[np.ndarray] | None]:
    """
    Load rollout data and images from a saved rollout directory.

    Args:
        rollout_path: Path to the rollout directory (saved by save_rollout).
        states_samples: Maximum number of states to sample uniformly.
        images_samples: Maximum number of images to sample uniformly.

    Returns:
        Tuple of (rollout_data, images) where each is None if not present.
        - rollout_data: List of state dictionaries (uniformly sampled)
        - images: List of numpy arrays (uniformly sampled frames from the GIF)
    """
    rollout_data = None
    images = None

    # Load rollout data from JSONL
    data_path = os.path.join(rollout_path, "rollout_data.jsonl")
    if os.path.exists(data_path):
        all_states = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    all_states.append(json.loads(line))
        rollout_data = uniform_sample(all_states, states_samples)

    # Load images from GIF
    gif_path = os.path.join(rollout_path, "rollout.gif")
    if os.path.exists(gif_path):
        all_frames = []
        gif = Image.open(gif_path)
        try:
            while True:
                frame = np.array(gif.convert("RGB"))
                all_frames.append(frame)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass  # End of frames
        images = uniform_sample(all_frames, images_samples)

    return rollout_data, images


def encode_images_to_base64(images: list[np.ndarray]) -> list[str]:
    """
    Encode a list of images (numpy arrays) to base64 strings.

    Args:
        images: List of images as numpy arrays.

    Returns:
        List of base64-encoded image strings.
    """
    import base64
    from io import BytesIO

    encoded_images = []
    for img_array in images:
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_images.append(img_str)

    return encoded_images


def _format_value(value: Any) -> str:
    """Format a single state value for LLM consumption."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
        return "[" + ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in value) + "]"
    return str(value)


def rollout_states_to_str(rollout_data: list[dict[str, Any]]) -> str:
    """
    Encode rollout into a string format suitable for LLM consumption.

    Each state dict must have:
      - "agents": dict mapping agent name → dict of agent fields
      - "env":    dict mapping env-level field name → value

    Args:
        rollout_data: List of state dictionaries.

    Returns:
        A formatted string representation of all rollout states.
    """
    if not rollout_data:
        return "Empty rollout"

    all_lines = []
    for state in rollout_data:
        state_idx = state.pop("step", "N/A")
        state_parts = []

        # Agents section
        for agent_name, agent_state in state.get("agents", {}).items():
            for key, value in agent_state.items():
                state_parts.append(f"{agent_name}.{key}={_format_value(value)}")

        # Env section
        for key, value in state.get("env", {}).items():
            state_parts.append(f"env.{key}={_format_value(value)}")

        all_lines.append(f"Step {state_idx}: " + ", ".join(state_parts))

    return "\n".join(all_lines).rstrip()
