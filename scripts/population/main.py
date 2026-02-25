import os
import json
import importlib

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from popgen.population import (
    RLAgentGenerationPipeline,
    PolicyCodeAgentGenerationPipeline,
    LLMAgentGenerationPipeline,
    RewardFnGenerator,
    PolicyCodeGenerator,
    BehaviorAdherenceEvaluator,
    Behavior,
)
from popgen.utils.config import LitellmConfig

OmegaConf.register_new_resolver("basename", lambda s: s.split("/")[-1])
OmegaConf.register_new_resolver("classname", lambda s: s.split(".")[-1])
OmegaConf.register_new_resolver(
    "behavior_source",
    lambda bp, bt: os.path.splitext(os.path.basename(str(bp)))[0] if bp else bt.split(".")[-1],
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()


def _parse_trailing_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _next_trial_index(output_dir: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    existing = []
    for name in os.listdir(output_dir):
        idx = _parse_trailing_index(name, "trial_")
        if idx is not None:
            existing.append(idx)
    return (max(existing) + 1) if existing else 0


def run_pipeline(cfg: DictConfig):
    generation_type = cfg.generation.type

    run_name = cfg.run_name
    extended_output_dir = os.path.join(cfg.output_dir, run_name)
    behavior_file_path = os.path.join(extended_output_dir, "behaviors.json")
    os.makedirs(extended_output_dir, exist_ok=True)

    behavior_path = cfg.get("behavior_path", None)

    logging.info("=" * 60)
    logging.info("POPULATION GENERATION PIPELINE")
    logging.info("=" * 60)
    logging.info(f"Generation:  {generation_type}")
    if behavior_path:
        logging.info(f"Behavior:    [from file] {behavior_path}")
    else:
        logging.info(f"Behavior:    {cfg.behavior._target_}")
    logging.info(f"Environment: {cfg.env}")
    logging.info(f"Output:      {extended_output_dir}")
    logging.info("-" * 60)

    env = importlib.import_module(f"popgen.envs.{cfg.env}")

    # ---- Stage 1: Behavior loading/generation ----

    logging.info("")
    logging.info("STAGE 1: Behavior Generation")
    logging.info("-" * 40)

    if behavior_path:
        logging.info(f"Loading behaviors from {behavior_path}")
        with open(behavior_path, "r") as f:
            behaviors_data = json.load(f)
        behaviors = [Behavior.model_validate(b) for b in behaviors_data]
        # Save a copy to the output dir for completeness
        with open(behavior_file_path, "w") as f:
            json.dump(behaviors_data, f, indent=2)
        logging.info(f"Saved behavior copy to {behavior_file_path}")
    else:
        behavior_generator = instantiate(cfg.behavior, output_type=Behavior)

        if os.path.exists(behavior_file_path):
            logging.info(f"Loading cached behaviors from {behavior_file_path}")
            with open(behavior_file_path, "r") as f:
                behaviors_data = json.load(f)
            behaviors = [Behavior.model_validate(b) for b in behaviors_data]
        else:
            logging.info("Generating new behaviors...")
            try:
                behaviors = behavior_generator.generate(
                    input_prompt=env.ENVIRONMENT_DESCRIPTION,
                    num_items=cfg.n_behaviors,
                )
                behaviors_data = [b.model_dump() for b in behaviors]
                with open(behavior_file_path, "w") as f:
                    json.dump(behaviors_data, f, indent=2)
                logging.info(f"Saved behaviors to {behavior_file_path}")
            except Exception as e:
                logging.error(f"Behavior generation failed: {e}")
                return

    assert len(behaviors) > 0, "No behaviors loaded."

    # ensure we only have the correct number of behaviors (in case behavior generator returned more than requested)
    behaviors = behaviors[:cfg.n_behaviors]

    logging.info(f"Loaded {len(behaviors)} behaviors:")
    for i, b in enumerate(behaviors):
        logging.info(f"  [{i}] {b.name}")

    # ---- Stage 2: Generation-specific pipeline (repeated for each trial) ----

    num_trials = cfg.get("num_trials", 1)
    trial_start_idx = _next_trial_index(extended_output_dir) if num_trials > 1 else 0
    if num_trials > 1 and trial_start_idx > 0:
        logging.info(f"Found existing trials. Appending {num_trials} new trial(s) starting at trial_{trial_start_idx}.")

    for trial_i in range(num_trials):
        trial_idx = trial_start_idx + trial_i
        trial_output_dir = os.path.join(extended_output_dir, f"trial_{trial_idx}")
        os.makedirs(trial_output_dir, exist_ok=True)
        if num_trials > 1:
            logging.info("")
            logging.info(f"TRIAL {trial_idx} ({trial_i + 1}/{num_trials} in this run)")
            logging.info("-" * 40)

        trial_seed = cfg.seed + trial_idx
        if generation_type == "rl_reward":
            _run_rl_reward(cfg, behaviors, env, trial_output_dir, seed=trial_seed)
        elif generation_type == "policy_code":
            _run_policy_code(cfg, behaviors, env, trial_output_dir, seed=trial_seed)
        elif generation_type == "llm_agent":
            _run_llm_agent(cfg, behaviors, env, trial_output_dir, seed=trial_seed)
        else:
            raise ValueError(f"Unknown generation type: '{generation_type}'. Choose from: rl_reward, policy_code, llm_agent")


def _run_rl_reward(cfg, behaviors, env, output_dir, seed=None):
    if not cfg.generation.get("align", True):
        logging.info("")
        logging.info("=" * 60)
        logging.info("STAGE 1 COMPLETE (align=False, skipping RL training)")
        logging.info("=" * 60)
        return

    logging.info("")
    logging.info("STAGE 2: Reward Generation, RL Training & Evaluation")
    logging.info(f"Processing {len(behaviors)} behaviors with {cfg.n_workers} worker(s)")
    logging.info("-" * 40)

    reward_generator = RewardFnGenerator(
        model_config=LitellmConfig(model_name=cfg.generation.model_name)
    )
    evaluator = BehaviorAdherenceEvaluator(
        model_config=LitellmConfig(model_name=cfg.evaluation.evaluator_model)
    )

    seed = seed if seed is not None else cfg.seed
    pipeline = RLAgentGenerationPipeline(
        generator=reward_generator,
        evaluator=evaluator,
        env_module=env,
        output_dir=output_dir,
        training_attempts=cfg.generation.get("training_attempts", 1),
    )
    successes, summary = pipeline(
        behaviors=behaviors,
        num_attempts=cfg.evaluation.max_alignment_retries,
        n_rollouts=cfg.evaluation.n_eval_rollouts,
        adherence_threshold=cfg.evaluation.adherence_threshold,
        num_workers=cfg.n_workers,
        evaluation_kwargs={"seed": seed},
    )
    _log_summary(behaviors, successes, summary)


def _run_policy_code(cfg, behaviors, env, output_dir, seed=None):
    logging.info("")
    logging.info("STAGE 2: Policy Code Generation & Evaluation")
    logging.info(f"Processing {len(behaviors)} behaviors with {cfg.n_workers} worker(s)")
    logging.info("-" * 40)

    policy_generator = PolicyCodeGenerator(
        model_config=LitellmConfig(model_name=cfg.generation.model_name)
    )
    evaluator = BehaviorAdherenceEvaluator(
        model_config=LitellmConfig(model_name=cfg.evaluation.evaluator_model)
    )

    seed = seed if seed is not None else cfg.seed

    pipeline = PolicyCodeAgentGenerationPipeline(
        generator=policy_generator,
        evaluator=evaluator,
        env_module=env,
        output_dir=output_dir,
    )
    successes, summary = pipeline(
        behaviors=behaviors,
        num_attempts=cfg.evaluation.max_alignment_retries,
        n_rollouts=cfg.evaluation.n_eval_rollouts,
        adherence_threshold=cfg.evaluation.adherence_threshold,
        num_workers=cfg.n_workers,
        evaluation_kwargs={"seed": seed},
    )
    _log_summary(behaviors, successes, summary)


def _run_llm_agent(cfg, behaviors, env, output_dir, seed=None):
    if not cfg.generation.get("evaluate", True):
        logging.info("")
        logging.info("=" * 60)
        logging.info("STAGE 1 COMPLETE (evaluate=False, skipping agent evaluation)")
        logging.info("=" * 60)
        return

    logging.info("")
    logging.info("STAGE 2: LLM Agent Configuration & Evaluation")
    logging.info(f"Processing {len(behaviors)} behaviors with {cfg.n_workers} worker(s)")
    logging.info("-" * 40)

    agent_config = LitellmConfig(model_name=cfg.generation.model_name)
    evaluator = BehaviorAdherenceEvaluator(
        model_config=LitellmConfig(model_name=cfg.evaluation.evaluator_model)
    )

    _reserved_keys = {"type", "evaluate", "model_name", "max_evaluation_attempts"}
    agent_kwargs = {k: v for k, v in cfg.generation.items() if k not in _reserved_keys}

    seed = seed if seed is not None else cfg.seed
    pipeline = LLMAgentGenerationPipeline(
        agent_model_config=agent_config,
        evaluator=evaluator,
        env_module=env,
        output_dir=output_dir,
    )
    successes, summary = pipeline(
        behaviors=behaviors,
        num_attempts=cfg.generation.get("max_evaluation_attempts", cfg.evaluation.max_alignment_retries),
        n_rollouts=cfg.evaluation.n_eval_rollouts,
        adherence_threshold=cfg.evaluation.adherence_threshold,
        num_workers=cfg.n_workers,
        generation_kwargs=agent_kwargs,
        evaluation_kwargs={"seed": seed},
    )
    _log_summary(behaviors, successes, summary)


def _log_summary(behaviors, successes, summary):
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
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
