import os
import sys
import json
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from popgen.population import (
    FeedbackGenerationPipeline,
    InContextStrategy,
    StratifiedStrategy,
    PolicyGenerator,
    BehaviorEvaluator,
    Behavior,
)
from popgen.utils.logging import save_rollout
from popgen.utils.imports import import_function_from_file

# Set log levels
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# load env variables from .env file
from dotenv import load_dotenv
load_dotenv()


def run_behavior_pipeline(cfg: DictConfig):
    """Runs the full behavior generation, policy generation, rollout, and evaluation pipeline."""
    # ----------- Configuration ----------- #

    logging.info("=" * 60)
    logging.info("BEHAVIOR GENERATION PIPELINE")
    logging.info("=" * 60)
    logging.info(f"Strategy:    {cfg.strategy}")
    logging.info(f"LLM Model:   {cfg.llm_model}")
    logging.info(f"Environment: {cfg.env.name}")
    logging.info(f"Output:      {cfg.output_dir}")
    logging.info("-" * 60)

    # Create output directories
    run_id = f"{cfg.env.name}__{cfg.strategy}__{cfg.llm_model.split('/')[-1]}"
    extended_output_dir = os.path.join(cfg.output_dir, run_id)
    behavior_file_path = os.path.join(extended_output_dir, "behaviors.json")
    os.makedirs(extended_output_dir, exist_ok=True)

    # Load env
    env = importlib.import_module(cfg.env.env_path)
    env_description = env.ENVIRONMENT_DESCRIPTION
    observation_format = env.OBSERVATION_FORMAT
    rollout_fn = env.rollout_behavior

    # Initialize generators
    if cfg.strategy == "in_context":
        behavior_generator = InContextStrategy(model=cfg.llm_model, output_type=Behavior)
    elif cfg.strategy == "stratified":
        behavior_generator = StratifiedStrategy(model=cfg.llm_model, output_type=Behavior)
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")
    policy_generator = PolicyGenerator(model=cfg.llm_model)
    policy_generator_kwargs = {"observation_format": observation_format}
    evaluator = BehaviorEvaluator(model=cfg.llm_model)
    agent_generation_pipeline = FeedbackGenerationPipeline(
        generator=policy_generator,
        evaluator=evaluator,
        rollout_fn=rollout_fn,
        output_dir=extended_output_dir,
    )

    # ------------------ Generate Behavior ------------- #

    logging.info("")
    logging.info("STAGE 1: Behavior Generation")
    logging.info("-" * 40)

    # if behaviors already exist, load them
    if os.path.exists(behavior_file_path):
        logging.info(f"Loading cached behaviors from {behavior_file_path}")
        with open(behavior_file_path, "r") as f:
            behaviors_data = json.load(f)
        behaviors = [Behavior.model_validate(b) for b in behaviors_data]
    else:
        logging.info("Generating new behaviors...")
        try:
            behaviors = behavior_generator.generate(
                environment_description=env_description
            )

            # Save as list of dicts for compatibility
            behaviors_data = [b.model_dump() for b in behaviors]
            with open(behavior_file_path, "w") as f:
                json.dump(behaviors_data, f, indent=2)
            logging.info(f"Saved behaviors to {behavior_file_path}")

        except Exception as e:
            logging.error(f"Behavior generation failed: {e}")
            return

    assert len(behaviors) > 0, "No behaviors generated."

    logging.info(f"Generated {len(behaviors)} behaviors:")
    for i, b in enumerate(behaviors):
        logging.info(f"  [{i}] {b.name}")

    # ----------------- Policy Generation & Evaluation ------------ #

    logging.info("")
    logging.info("STAGE 2: Policy Generation & Evaluation")
    logging.info("-" * 40)

    results_summary = []

    for idx, behavior in enumerate(behaviors):
        logging.info(f"Processing Behavior [{idx}]: {behavior.name}")
        success, eval_data = agent_generation_pipeline(
            behavior=behavior,
            behavior_index=idx,
            num_attempts=cfg.max_alignment_retries,
            **policy_generator_kwargs
        )

        results_summary.append({
            "behavior": behavior.name,
            "success": success,
            "attempts": idx + 1,
            "details": eval_data,
        })

    # ----------------- Summary ------------ #

    logging.info("")
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)

    successful = sum(1 for r in results_summary if r["success"])
    logging.info(f"Results: {successful}/{len(results_summary)} behaviors aligned")

    for r in results_summary:
        status = "PASS" if r["success"] else "FAIL"
        logging.info(f"  [{status}] {r['behavior']} (attempts: {r['attempts']})")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_behavior_pipeline(cfg)

if __name__ == "__main__":
    main()
