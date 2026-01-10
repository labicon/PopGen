import os
import sys
import json
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

from popgen.population.generation import (
    InContextStrategy,
    StratifiedStrategy,
    PolicyGenerator,
    Behavior,
)
from popgen.population.evaluation import BehaviorEvaluator
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


def _log_prefix(behavior_idx: int = None, align_idx: int = None, rollout_idx: int = None) -> str:
    """Generate a structured log prefix like [B=0 | A=1 | R=2]."""
    parts = []
    if behavior_idx is not None:
        parts.append(f"B={behavior_idx}")
    if align_idx is not None:
        parts.append(f"A={align_idx}")
    if rollout_idx is not None:
        parts.append(f"R={rollout_idx}")
    return f"[{' | '.join(parts)}]" if parts else ""


def generate_policy_rollout_evaluate(
    generator: PolicyGenerator,
    evaluator: BehaviorEvaluator,
    behavior: Behavior,
    behavior_index: int,
    observation_format: str,
    rollout_fn,
    output_dir: str,
    alignment_attempt: int = 0,
    previous_policy: str = None,
    error_message: str = None,
    max_retries: int = 3,
):
    """
    Generates policy, runs rollout, and evaluates the result.
    Returns (success, result) tuple.
    """
    prefix = _log_prefix(behavior_index, alignment_attempt)
    rollout_success = False

    output_dir = os.path.join(
        output_dir, f"behavior_{behavior_index}", f"attempt_{alignment_attempt}"
    )

    policy_code = None

    for attempt in range(max_retries):
        r_prefix = _log_prefix(behavior_index, alignment_attempt, attempt)
        logging.info(f"{r_prefix} Generating policy...")

        rollout_output_dir = f"{output_dir}/rollout_{attempt}"
        os.makedirs(rollout_output_dir, exist_ok=True)

        # Generate Policy
        try:
            policy_code_path = os.path.join(rollout_output_dir, f"policy.py")

            # Generate agent0 policy
            agent0_policy = generator.generate_policy(
                behavior=behavior,
                observation_format=observation_format,
                agent_id=0,
                previous_policy=previous_policy,
                error_message=error_message,
            )

            # Generate agent1 policy that coordinates with agent0
            agent1_policy = generator.generate_policy(
                behavior=behavior,
                observation_format=observation_format,
                agent_id=1,
                partner_behavior=behavior,
                partner_policy=agent0_policy.code,
            )

            # Combine policies
            policy_code = agent0_policy.code + "\n\n" + agent1_policy.code
            logging.debug(f"{r_prefix} Generated policy:\n{policy_code}")

            # Save to file
            with open(policy_code_path, "w") as f:
                f.write(policy_code)
            logging.debug(f"{r_prefix} Saved policy to {policy_code_path}")

        except Exception as e:
            logging.error(f"{r_prefix} Policy generation failed: {e}")
            if attempt == max_retries - 1:
                return False, {
                    "reasoning": f"Policy generation failed after {max_retries} attempts: {e}"
                }
            continue

        # Run Rollout
        try:
            logging.info(f"{r_prefix} Running rollout...")

            # TODO: Why is this needed?
            # Import the module
            # We need to reload the module to pick up the new policy if it was regenerated
            if "rollout_module" in sys.modules:
                del sys.modules["rollout_module"]

            # Force reload of the policy module by removing it from sys.modules
            # This is crucial because the policy file has been overwritten
            if "policy_rollout.envs.agent0_policy" in sys.modules:
                logging.debug(f"{r_prefix} Reloading policy module...")
                del sys.modules["policy_rollout.envs.agent0_policy"]

            agent0_policy_fn = import_function_from_file(policy_code_path, "agent0_policy")
            agent1_policy_fn = import_function_from_file(policy_code_path, "agent1_policy")

            # Call the rollout function
            rollout_data, images, error = rollout_fn(agent0_policy_fn, agent1_policy_fn)

            # Save rollout trajectory and images
            save_rollout(
                rollout_data=rollout_data,
                images=images,
                save_path=rollout_output_dir,
                num_states=20   # TODO: parameterize num_states
            )
            logging.debug(f"{r_prefix} Saved rollout to {rollout_output_dir}")

            if error is not None:
                raise RuntimeError(error)
            else:
                logging.info(f"{r_prefix} Rollout successful")
                rollout_success = True
                break  # Success, exit retry loop

        except Exception as e:
            logging.error(f"{r_prefix} Rollout failed: {e}")
            error_message = str(e)
            previous_policy = policy_code

            if attempt == max_retries - 1:
                logging.error(f"{r_prefix} Max rollout retries reached")
                return False, {
                    "reasoning": f"Rollout failed after {max_retries} attempts: {e}"
                }
            logging.warning(f"{r_prefix} Retrying with error feedback...")

    if not rollout_success:
        return False, {"reasoning": "Rollout failed", "policy_code": policy_code}

    # 4. Evaluate Rollout
    logging.info(f"{prefix} Evaluating rollout...")

    # trajectory is str representation of rollout_data
    trajectory = "\n".join(str(state) for state in rollout_data)
    images_path = [f"{rollout_output_dir}/rollout.gif"]

    # Evaluate
    result = evaluator.evaluate_trajectory_and_images(
        trajectory,
        images_path,
        behavior
    )

    # Log evaluation result
    log_file_path = f"{output_dir}/evaluation.json"
    log_data = {
        "behavior": behavior.model_dump(),
        "trajectory": trajectory,
        "evaluation_result": result.model_dump(),
    }
    with open(log_file_path, "w") as f:
        json.dump(log_data, f, indent=2)

    if result.evaluation:
        logging.info(f"{prefix} Evaluation PASSED")
    else:
        logging.warning(f"{prefix} Evaluation FAILED: {result.reasoning[:100]}...")

    return result.evaluation, {
        "evaluation": result.evaluation,
        "reasoning": result.reasoning,
        "policy_code": policy_code,
    }


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
    policy_generator = PolicyGenerator(model=cfg.llm_model)
    if cfg.strategy == "in_context":
        behavior_generator = InContextStrategy(model=cfg.llm_model)
    elif cfg.strategy == "stratified":
        behavior_generator = StratifiedStrategy(model=cfg.llm_model)
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")
    evaluator = BehaviorEvaluator(model=cfg.llm_model)

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
        b_prefix = _log_prefix(idx)
        logging.info("")
        logging.info(f"{b_prefix} Processing: {behavior.name}")
        logging.info(f"{b_prefix} Description: {behavior.description[:80]}...")

        # Loop until policy is aligned with behavior
        max_alignment_retries = cfg.max_alignment_retries
        previous_policy = None
        error_message = None
        success = False

        for i in range(max_alignment_retries):
            a_prefix = _log_prefix(idx, i)
            logging.info(f"{a_prefix} Starting alignment attempt ({i+1}/{max_alignment_retries})")

            success, result = generate_policy_rollout_evaluate(
                generator=policy_generator,
                evaluator=evaluator,
                behavior=behavior,
                behavior_index=idx,
                output_dir=extended_output_dir,
                rollout_fn=rollout_fn,
                observation_format=observation_format,
                alignment_attempt=i,
                previous_policy=previous_policy,
                error_message=error_message,
                max_retries=cfg.max_rollout_retries,
            )

            if success:
                logging.info(f"{a_prefix} SUCCESS - Policy aligned with behavior")
                break
            else:
                error_message = result.get("reasoning", "Unknown")
                previous_policy = result.get("policy_code", None)

                if i == max_alignment_retries - 1:
                    logging.error(f"{a_prefix} FAILED - Max alignment retries exhausted")

        results_summary.append({
            "behavior": behavior.name,
            "success": success,
            "attempts": i + 1
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_behavior_pipeline(cfg)

if __name__ == "__main__":
    main()
