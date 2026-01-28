import json
import logging
import os
from typing import Any, Optional

from popgen.population.pipeline import load_policy_and_env
from popgen.utils.logging import save_rollout, uniform_sample

logger = logging.getLogger(__name__)


def run_rollout(
    rollout_fn: callable,
    output_dir: str,
    n_rollouts: int = 5,
    subdir_prefix: str = "rollout",
) -> tuple[list, list]:
    """Run multiple rollouts and collect sampled states and images.

    Ref: popgen/population/rl_pipeline.py
    """
    all_rollouts_states = []
    all_rollouts_images = []

    logger.info(f"Loading env and policy from {output_dir}")
    try:
        artifacts_dict = load_policy_and_env(policy_type="RewardFn", save_dir=output_dir)
        env = artifacts_dict["env"]
        policies = artifacts_dict["policies"]
    except Exception as e:
        logger.error(f"Failed to load policy/env: {e}")
        return [], []

    logger.info(f"Starting {n_rollouts} rollouts in {output_dir}")

    for i in range(n_rollouts):
        logger.info(f"Running rollout ({i + 1}/{n_rollouts})...")
        rollout_subdir = os.path.join(output_dir, f"{subdir_prefix}_{i}")

        try:
            rollout_data, images, error = rollout_fn(**policies, env=env)
        except Exception as e:
            logger.error(f"Rollout execution exception: {e}")
            continue

        if error:
            logger.error(f"Rollout error: {error}")
            continue

        # Save rollout trajectory and images
        save_rollout(
            rollout_data=rollout_data,
            images=images,
            save_path=rollout_subdir,
        )
        logger.debug(f"Saved rollout to {rollout_subdir}")

        # Uniform sample for analysis
        # Using same sampling as RLPipeline
        all_rollouts_states.append(uniform_sample(rollout_data, n_samples=20))
        all_rollouts_images.append(uniform_sample(images, n_samples=5))

    return all_rollouts_states, all_rollouts_images


def find_successful_attempt(behavior_path: str, min_adherence: float = 0.6) -> tuple[str | None, dict | None]:
    """
    Finds the first attempt in behavior_path with adherence > min_adherence.
    Returns (attempt_path, evaluation_data) or (None, None).
    """
    if not os.path.isdir(behavior_path):
        return None, None

    try:
        attempt_dirs = [d for d in os.listdir(behavior_path) if d.startswith("attempt_")]
    except OSError:
        return None, None

    if not attempt_dirs:
        return None, None

    # Sort attempts by index
    try:
        attempt_dirs.sort(key=lambda x: int(x.split("_")[1]))
    except ValueError:
        # Fallback if naming is weird
        attempt_dirs.sort()

    for attempt in attempt_dirs:
        a_path = os.path.join(behavior_path, attempt)
        eval_path = os.path.join(a_path, "evaluation.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    eval_data = json.load(f)

                # Check metrics
                metrics = eval_data.get("overall_metrics", {})
                if metrics.get("adherence", 0.0) > min_adherence:
                    return a_path, eval_data
            except Exception as e:
                logger.warning(f"Error reading {eval_path}: {e}")
                continue

    return None, None
