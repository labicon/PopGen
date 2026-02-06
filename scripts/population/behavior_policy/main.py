import os
import json
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf
from popgen.population import (
    PolicyCodeAgentGenerationPipeline,
    InContextStrategy,
    StratifiedStrategy,
    PolicyCodeGenerator,
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
    """Runs the full behavior generation, policy generation, rollout, and evaluation pipeline."""
    # ----------- Configuration ----------- #

    # Create output directories
    run_name = cfg.run_name
    extended_output_dir = os.path.join(cfg.output_dir, run_name)
    behavior_file_path = os.path.join(extended_output_dir, "behaviors.json")
    os.makedirs(extended_output_dir, exist_ok=True)

    logging.info("=" * 60)
    logging.info("BEHAVIOR GENERATION PIPELINE")
    logging.info("=" * 60)
    logging.info(f"Strategy:    {cfg.behavior_generator.strategy}")
    logging.info(f"Generator:   {cfg.generator.model_name}")
    logging.info(f"Environment: {cfg.env}")
    logging.info(f"Output:      {extended_output_dir}")
    logging.info("-" * 60)

    # Load env
    env = importlib.import_module(f"popgen.envs.{cfg.env}")
    env_description = env.ENVIRONMENT_DESCRIPTION
    observation_format = env.OBSERVATION_FORMAT
    action_format = env.ACTION_FORMAT
    ego_visual_description = env.EGO_VISUAL_DESCRIPTION
    rollout_fn = env.rollout_fn

    policy_generator_kwargs = {
        "observation_format": observation_format,
        "action_format": action_format,
    }
    evaluation_kwargs = {
        "ego_visual_description": ego_visual_description,
        "ego_name": "agent0",
    }

    # ------------------ Generate Behavior ------------- #

    logging.info("")
    logging.info("STAGE 1: Behavior Generation")
    logging.info("-" * 40)

    # Initialize generator
    behavior_strategy = cfg.behavior_generator.strategy
    behavior_model_config = LitellmConfig(**cfg.behavior_generator.model_config)
    behavior_generator = load_strategy(behavior_strategy, model_config=behavior_model_config, output_type=Behavior)

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

    # ----------------- Policy Generation & Evaluation ------------ #

    logging.info("")
    logging.info("STAGE 2: Policy Generation & Evaluation")
    logging.info(f"Processing {len(behaviors)} behaviors with {cfg.n_workers} worker(s)")
    logging.info("-" * 40)

    policy_generator = PolicyCodeGenerator(model_config=LitellmConfig(**cfg.generator))
    evaluator = BehaviorAdherenceEvaluator(model_config=LitellmConfig(**cfg.evaluator))
    pipeline = PolicyCodeAgentGenerationPipeline(
        generator=policy_generator,
        evaluator=evaluator,
        rollout_fn=rollout_fn,
        output_dir=extended_output_dir,
    )

    # Run pipeline on all behaviors
    successes, summary = pipeline(
        behaviors=behaviors,
        num_attempts=cfg.max_alignment_retries,
        n_rollouts=cfg.n_eval_rollouts,
        adherence_threshold=cfg.adherence_threshold,
        num_workers=cfg.n_workers,
        generation_kwargs=policy_generator_kwargs,
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
