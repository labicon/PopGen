import os
import base64
import io
from PIL import Image

from typing import Any, Dict, List, Union

def encode_image(image_path: str) -> Union[str, List[str]]:
    """Encode an image or directory of images to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_gif(gif_path: str, num_frames: int = 5) -> List[str]:
    """Extract and encode frames from a GIF."""
    encoded_frames = []
    try:
        with Image.open(gif_path) as im:
            total_frames = im.n_frames
            step = max(1, total_frames // num_frames)
            for i in range(0, total_frames, step):
                im.seek(i)
                frame = im.convert("RGB")
                buffered = io.BytesIO()
                frame.save(buffered, format="JPEG")
                encoded_frames.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
                if len(encoded_frames) >= num_frames:
                    break
    except Exception as e:
        print(f"Error encoding GIF {gif_path}: {e}")
    return encoded_frames
