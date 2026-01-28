import json
import logging
import os
from typing import Any, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

from popgen.population.evaluation import SimilarityEvaluator

from .base_experiment import BaseAnalysisExperiment
from .visualization_utils import plot_similarity_comparison

logger = logging.getLogger(__name__)


# --- QwenEmbedder helper class ---
class QwenEmbedder:
    """Wrapper for Qwen-VL-Embedding models using HuggingFace Transformers."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing QwenEmbedder with {model_name} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=True, dtype="auto")
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Basic text encoding - using a space as placeholder if text is empty
            t = text if text else " "
            inputs = self.processor(text=[t], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                # Request hidden states
                outputs = self.model(**inputs, output_hidden_states=True)

                # Check for hidden_states
                if hasattr(outputs, "hidden_states"):
                    last_layer = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                    # Take last token
                    emb = last_layer[0, -1, :].float().cpu().numpy()
                elif hasattr(outputs, "last_hidden_state"):
                    # In case it returns BaseModelOutput
                    emb = outputs.last_hidden_state[0, -1, :].float().cpu().numpy()
                else:
                    raise ValueError("Model output has no hidden_states or last_hidden_state")

            embeddings.append(emb)

        return np.array(embeddings)

    def encode_video(self, frames: list[Image.Image]) -> np.ndarray:
        if not frames:
            return np.zeros(self.model.config.hidden_size)

        # Prepare inputs with empty text prompt
        text_prompt = " "
        inputs = self.processor(text=[text_prompt], images=frames, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            if hasattr(outputs, "hidden_states"):
                last_layer = outputs.hidden_states[-1]
                emb = last_layer[0, -1, :].float().cpu().numpy()
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[0, -1, :].float().cpu().numpy()
            else:
                raise ValueError("Model output has no hidden_states or last_hidden_state")

        return emb


class VideoSimilarityExperiment(BaseAnalysisExperiment):
    @property
    def experiment_name(self) -> str:
        return "video_similarity"

    def process_behavior(self, behavior_ind: int, behavior_data: dict, successful_attempt: str) -> str | None:
        """Returns the path to the video file, or None."""

        # Priority 1: analyze_rollout_0/rollout.gif
        p0 = os.path.join(successful_attempt, "analyze_rollout_0", "rollout.gif")
        if os.path.exists(p0):
            return p0

        # Priority 2: similarity_rollout_0/rollout.gif
        p1 = os.path.join(successful_attempt, "similarity_rollout_0", "rollout.gif")
        if os.path.exists(p1):
            return p1

        # Priority 3: diversity_rollout_0/rollout.gif
        p2 = os.path.join(successful_attempt, "diversity_rollout_0", "rollout.gif")
        if os.path.exists(p2):
            return p2

        # Priority 4: rollout.gif in attempt root
        p3 = os.path.join(successful_attempt, "rollout.gif")
        if os.path.exists(p3):
            return p3

        logger.warning(f"No video found in {successful_attempt}")
        return None

    def evaluate_population(self, desired_behaviors: list[dict], collected_data: list[str | None], env_name: str, strategy: str):
        # collected_data contains video paths
        video_paths = collected_data

        # Prepare text descriptions
        desired_texts = []
        behavior_names = []
        for i, desired in enumerate(desired_behaviors):
            d_desc = desired.get("description", "") if desired else ""
            desired_texts.append(d_desc)
            behavior_names.append(desired.get("name", f"Behavior {i}") if desired else f"Behavior {i}")

        embedding_model_name = self.cfg.get("embedding_model", "Qwen/Qwen3-VL-Embedding-8B")
        logger.info(f"Loading model {embedding_model_name}...")

        use_qwen = "Qwen" in embedding_model_name

        # 1. Encode Text
        if use_qwen:
            model_wrapper = QwenEmbedder(embedding_model_name)
            logger.info("Encoding desired behaviors (Qwen)...")
            text_embeddings = model_wrapper.encode_text(desired_texts)
        else:
            evaluator = SimilarityEvaluator(embedding_model=embedding_model_name)
            model = evaluator.model
            logger.info("Encoding desired behaviors (CLIP)...")
            text_embeddings = model.encode(desired_texts)

        # 2. Encode Videos
        logger.info("Encoding videos...")
        video_embeddings = []
        valid_mask = []

        for i, vp in enumerate(video_paths):
            if not vp or not os.path.exists(vp):
                valid_mask.append(False)
                video_embeddings.append(np.zeros(text_embeddings.shape[1]))
                continue

            try:
                im = Image.open(vp)
                frames = []
                try:
                    while True:
                        frames.append(im.copy().convert("RGB"))
                        im.seek(len(frames))
                except EOFError:
                    pass

                max_frames = 24 if use_qwen else 10
                if len(frames) > max_frames:
                    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                    frames = [frames[j] for j in indices]

                logger.info(f"Encoding video {vp} with {len(frames)} frames...")

                if use_qwen:
                    video_embedding = model_wrapper.encode_video(frames)
                else:
                    frame_embeddings = model.encode(frames)
                    video_embedding = np.mean(frame_embeddings, axis=0)

                video_embeddings.append(video_embedding)
                valid_mask.append(True)

            except Exception as e:
                logger.error(f"Failed to encode video {vp}: {e}")
                video_embeddings.append(np.zeros(text_embeddings.shape[1]))
                valid_mask.append(False)

        video_embeddings = np.array(video_embeddings)

        # 3. Compute Similarity
        similarity_matrix = cosine_similarity(text_embeddings, video_embeddings)

        diagonal_scores = []
        for i in range(len(desired_behaviors)):
            if valid_mask[i]:
                score = similarity_matrix[i, i]
                diagonal_scores.append(float(score))
            else:
                diagonal_scores.append(0.0)

        mean_similarity = float(np.mean(diagonal_scores) if diagonal_scores else 0.0)
        logger.info(f"Mean Video-Text Similarity: {mean_similarity:.4f}")

        # 4. Visualize
        plot_similarity_comparison(
            behavior_names=behavior_names,
            output_path=self.output_path,
            embeddings_desired=text_embeddings,
            embeddings_analyzed=video_embeddings,
            semantic_scores=similarity_matrix,
            lexical_scores=np.zeros((len(behavior_names), len(behavior_names))),
            semantic_mean=mean_similarity,
            lexical_mean=0.0,
            env_name=env_name,
            strategy=strategy,
        )

        # 5. Save Report
        final_report = {"metrics": {"video_text_similarity": mean_similarity, "scores": diagonal_scores}, "video_paths": video_paths}
        with open(os.path.join(self.experiment_root, "video_similarity_report.json"), "w") as f:
            json.dump(final_report, f, indent=2)


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup Loggers
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    exp = VideoSimilarityExperiment(cfg)
    exp.run()


if __name__ == "__main__":
    main()
