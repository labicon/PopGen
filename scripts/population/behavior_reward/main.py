import os
import json
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

from popgen.population import (
    RLAgentGenerationPipeline,
    RewardFnGenerator,
    BehaviorAdherenceEvaluator,
    Behavior,
)
from popgen.population.generation.diversity import load_strategy
from popgen.utils.config import LitellmConfig

OmegaConf.register_new_resolver(
    "basename",
    lambda s: s.split("/")[-1],
)

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
    """Runs the full behavior generation, reward generation, RL training, and evaluation pipeline."""
    # ----------- Configuration ----------- #

    # Create output directories
    run_name = cfg.get("run_name", f"{cfg.env}__{cfg.strategy}__{cfg.llm_model.split('/')[-1]}")
    extended_output_dir = os.path.join(cfg.output_dir, run_name)
    behavior_file_path = os.path.join(extended_output_dir, "behaviors.json")
    os.makedirs(extended_output_dir, exist_ok=True)

    logging.info("=" * 60)
    logging.info("RL AGENT GENERATION PIPELINE")
    logging.info("=" * 60)
    logging.info(f"Strategy:    {cfg.strategy}")
    logging.info(f"LLM Model:   {cfg.llm_model}")
    logging.info(f"Environment: {cfg.env}")
    logging.info(f"Align:       {cfg.align}")
    logging.info(f"Output:      {extended_output_dir}")
    logging.info("-" * 60)

    # Load env
    env = importlib.import_module(f"popgen.envs.{cfg.env}")
    env_description = env.ENVIRONMENT_DESCRIPTION
    reward_example = env.REWARD_EXAMPLE
    rollout_fn = env.rollout_fn
    train_fn = env.train_fn

    # Initialize generators
    llm_config = LitellmConfig(model_name=cfg.llm_model)
    behavior_generator = load_strategy(cfg.strategy, model_config=llm_config, output_type=Behavior)

    generation_kwargs = {
        "environment_description": env_description,
        "examples": [reward_example],
    }
    evaluator = BehaviorAdherenceEvaluator(model_config=llm_config)
    evaluation_kwargs = {
        "environment_description": env_description,
    }

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
                input_prompt=env_description
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

    if not cfg.align:
        logging.info("")
        logging.info("=" * 60)
        logging.info("STAGE 1 COMPLETE (align=False, skipping policy generation)")
        logging.info("=" * 60)
        return

    # ----------------- Reward Generation, RL Training & Evaluation ------------ #

    logging.info("")
    logging.info("STAGE 2: Reward Generation, RL Training & Evaluation")
    logging.info(f"Processing {len(behaviors)} behaviors with {cfg.n_workers} worker(s)")
    logging.info("-" * 40)

    reward_generator = RewardFnGenerator(model_config=llm_config)
    evaluator = BehaviorAdherenceEvaluator(model_config=llm_config)
    pipeline = RLAgentGenerationPipeline(
        generator=reward_generator,
        evaluator=evaluator,
        rollout_fn=rollout_fn,
        output_dir=extended_output_dir,
        train_fn=train_fn,
        training_attempts=cfg.get("training_attempts", 1),
    )

    # Run pipeline on all behaviors
    successes, summary = pipeline(
        behaviors=behaviors,
        num_attempts=cfg.max_alignment_retries,
        n_rollouts=cfg.n_eval_rollouts,
        adherence_threshold=cfg.adherence_threshold,
        num_workers=cfg.n_workers,
        generation_kwargs=generation_kwargs,
        evaluation_kwargs=evaluation_kwargs,
    )

    # ----------------- Summary ------------ #

    logging.info("")
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)

    logging.info(f"Results: {summary['n_succeeded']}/{summary['n_behaviors']} behaviors aligned")

    for idx, behavior in enumerate(behaviors):
        steps = summary["num_attempts_to_success"].get(str(idx))
        status = "PASS" if successes[idx] else "FAIL"
        attempts_str = f"(attempts: {steps})" if steps else ""
        logging.info(f"  [{status}] {behavior.name} {attempts_str}")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_behavior_pipeline(cfg)

if __name__ == "__main__":
    main()