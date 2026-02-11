import importlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any, Literal

import torch
from pqdm.processes import pqdm

from popgen.agents import PolicyCodeAgent
from popgen.utils.config import LitellmConfig
from popgen.utils.decorators import track_time
from popgen.utils.imports import import_all_functions_from_file, import_function_from_file, load_entrypoint
from popgen.utils.logging import save_rollout, uniform_sample

from .evaluation import BehaviorAdherenceEvaluator
from .generation import BaseGenerator
from .types import Behavior

logging.basicConfig(level=logging.INFO)


def load_policy_and_env(policy_type: Literal["PolicyCode", "RewardFn", "LLMAgent", "Trajectory"], **kwargs) -> dict:
    """Load policy functions and environment."""
    if policy_type == "PolicyCode":
        policies = import_all_functions_from_file(kwargs["file_path"])
        policies = {name: PolicyCodeAgent(func) for name, func in policies.items()}
        env = None  # Environment will be created during rollout
    elif policy_type == "RewardFn":
        from benchmarl.experiment import Experiment

        # Load experiment
        save_dir = Path(kwargs["save_dir"])
        checkpoints = sorted(list(save_dir.rglob("checkpoints/checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {save_dir}")

        checkpoint = checkpoints[-1]
        experiment_folder = checkpoint.parent.parent.resolve()
        experiment = Experiment.reload_from_file(
            str(checkpoint),
            experiment_patch={
                "save_folder": str(experiment_folder.parent),
                "loggers": [],
                "create_json": False,
            },
        )
        policies = {"policy": experiment.policy}
        env = experiment.test_env
    elif policy_type == "LLMAgent":
        from popgen.agents import LLMAgent
        from popgen.utils.config import LitellmConfig

        # Load configuration from JSON
        config_path = kwargs["config_path"]
        with open(config_path) as f:
            config = json.load(f)

        # Create LitellmConfig from the saved configuration
        model_config = LitellmConfig(**config["model_config"])

        # Load action type module
        action_type = load_entrypoint(config["action_type_module"])

        # Create LLMAgent with loaded configuration
        policy = LLMAgent(
            behavior=config["behavior"],
            model_config=model_config,
            environment_description=config["environment_description"],
            action_type=action_type,
            horizon_length=config.get("horizon_length", 1),
            include_trajectory_history=config.get("include_trajectory_history", False),
        )

        policies = {"policy": policy}
        env = None  # Environment will be created during rollout

    return {"env": env, "policies": policies}


def _worker_rollout(args: dict) -> dict:
    """Worker function for parallel rollout execution."""
    # Import environment module
    rollout_fn = args["rollout_fn"]
    artifacts_dict = load_policy_and_env(policy_type=args["policy_type"], **args.get("loading_kwargs", {}))

    # Perform rollout
    rollout_data, images, error = rollout_fn(**artifacts_dict["policies"], env=artifacts_dict["env"], **args["rollout_kwargs"])

    # Save rollout trajectory and images
    save_rollout(
        rollout_data=rollout_data,
        images=images,
        save_path=args["rollout_subfolder"],
    )

    if error is not None:
        return {"error": str(error), "states": None, "images": None}

    return {
        "error": None,
        "states": uniform_sample(rollout_data, n_samples=20),
        "images": uniform_sample(images, n_samples=5),
    }


class AgentGenerationPipeline(ABC):
    """Base pipeline for behavior-aligned policy generation and evaluation."""

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BehaviorAdherenceEvaluator,
        rollout_fn: callable,
        output_dir: str,
    ):
        self.generator: BaseGenerator = generator
        self.evaluator: BehaviorAdherenceEvaluator = evaluator
        self.rollout_fn = rollout_fn
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def _generate_policies(
        self,
        behaviors: list[Behavior],
        env_module: Any,
        feedbacks: list[list[tuple[str, str]]],
        output_dirs: list[str],
        num_workers: int = 1,
        **kwargs,
    ) -> tuple[list[Any], list[dict]]:
        """Generate policies for a batch of behaviors.

        Returns:
            Tuple of (generated_policies, loading_kwargs_list)
        """
        pass

    def __call__(
        self,
        behaviors: list[Behavior],
        num_attempts: int = 5,
        n_rollouts: int = 3,
        adherence_threshold: float = 1.0,
        num_workers: int = 8,
        generation_kwargs: dict = {},
        evaluation_kwargs: dict = {},
    ) -> tuple[list[bool], dict[str, Any]]:
        """Run the pipeline for a list of behaviors.

        Args:
            behaviors: List of Behavior objects
            num_attempts: Maximum number of alignment attempts per behavior
            n_rollouts: Number of rollouts per behavior for evaluation
            adherence_threshold: Minimum adherence score for success
            num_workers: Number of parallel workers for generation and rollouts
            generation_kwargs: Additional kwargs for generation
            evaluation_kwargs: Additional kwargs for evaluation

        Returns:
            Tuple of (success_flags, overall_summary)
        """
        start_time = time()

        n_behaviors = len(behaviors)

        # Track state per behavior
        feedbacks: list[list[tuple[str, str]]] = [[] for _ in range(n_behaviors)]
        generated_items: list[Any] = [None] * n_behaviors
        successes: list[bool] = [False] * n_behaviors
        attempts_to_success: list[int] = [None] * n_behaviors

        # Track per-attempt summaries
        attempt_summaries: list[dict[str, Any]] = []

        for attempt in range(num_attempts):
            # Get indices of behaviors that haven't succeeded yet
            pending_indices = [i for i in range(n_behaviors) if not successes[i]]
            if not pending_indices:
                break

            logging.info(f"[Attempt {attempt + 1}/{num_attempts}] Processing {len(pending_indices)} pending behaviors...")

            # Step 1: Generate policies for pending behaviors
            pending_behaviors = [behaviors[i] for i in pending_indices]
            pending_feedbacks = [feedbacks[i][-1:] for i in pending_indices]
            # make output dirs
            pending_output_dirs = []
            for i in pending_indices:
                dir = os.path.join(self.output_dir, f"behavior_{i}", f"attempt_{attempt}")
                pending_output_dirs.append(dir)
                os.makedirs(dir, exist_ok=True)

            (gen_results, loading_policy_kwargs), gen_latency = self._generate_policies(
                behaviors=pending_behaviors,
                feedbacks=pending_feedbacks,
                output_dirs=pending_output_dirs,
                num_workers=num_workers,
                **generation_kwargs,
            )

            # Map generated results back to original behavior indices
            # and filter out failed generations
            valid_indices = []
            valid_behaviors = []
            valid_loading_kwargs = []
            for i, idx in enumerate(pending_indices):
                # If gen_item is an error, skip rollout and evaluation
                if isinstance(gen_results[i], Exception) or gen_results[i] is None:
                    # Generation/training failed - add feedback for retry
                    logging.warning(f"[B={idx}] Generation/training failed: {gen_results[i]}")
                    feedbacks[idx].append(("", f"Generation or training failed: {gen_results[i]}. Please try again."))
                else:
                    generated_items[idx] = gen_results[i]
                    valid_indices.append(idx)
                    valid_behaviors.append(behaviors[idx])
                    valid_loading_kwargs.append(loading_policy_kwargs[i])

            # Step 2 & 3: Run rollouts and evaluate only for valid generations
            if valid_indices:
                valid_behavior_dirs = [os.path.join(self.output_dir, f"behavior_{idx}", f"attempt_{attempt}") for idx in valid_indices]
                behaviors_rollouts, rollout_latency = self._run_rollouts(
                    n_rollouts=n_rollouts,
                    attempt=attempt,
                    num_workers=num_workers,
                    loading_kwargs_list=valid_loading_kwargs,
                    behavior_dirs=valid_behavior_dirs,
                    **evaluation_kwargs,
                )
                eval_results, evaluation_latency = self._evaluate_policies(
                    behaviors=valid_behaviors,
                    behaviors_rollouts=behaviors_rollouts,
                    behavior_dirs=valid_behavior_dirs,
                    **evaluation_kwargs,
                )

                # Process evaluation results for valid indices
                for i, idx in enumerate(valid_indices):
                    instance_metric = eval_results.instance_metrics[i]
                    adherence = instance_metric["adherence"]
                    success = adherence >= adherence_threshold

                    # Map results back to original behavior index
                    successes[idx] = success

                    if success:
                        attempts_to_success[idx] = attempt + 1
                        logging.info(f"[B={idx}] SUCCESS on attempt {attempt + 1}")
                    else:
                        logging.warning(f"[B={idx}] FAILURE – Adherence: {adherence:.2f}")
                        overall_feedback = instance_metric["reasoning"]
                        # For code-based pipelines, include the generated code in feedback
                        # For other pipelines (e.g., LLM agents), just use the feedback text
                        if hasattr(generated_items[idx], "code"):
                            feedbacks[idx].append((generated_items[idx].code, overall_feedback))
                        else:
                            feedbacks[idx].append(("", overall_feedback))
            else:
                rollout_latency = 0.0
                evaluation_latency = 0.0

            # Log per-attempt summary
            attempt_summary = {
                "attempt": attempt + 1,
                "n_pending": len(pending_indices),
                "n_succeeded": sum(successes),
                "n_failed": len(pending_indices) - sum(successes),
                "latencies": {
                    "generation": gen_latency,
                    "rollout": rollout_latency,
                    "evaluation": evaluation_latency,
                },
                "results": dict(enumerate(successes)),
            }
            attempt_summaries.append(attempt_summary)
            logging.info(
                f"[Attempt {attempt + 1}] Gen: {gen_latency:.1f}s, Rollout: {rollout_latency:.1f}s, Eval: {evaluation_latency:.1f}s, Success: {attempt_summary['n_succeeded']}/{len(pending_indices)}"
            )

        # Compute overall summary
        n_attempts = len(attempt_summaries)
        total_time = time() - start_time

        overall_summary = {
            "n_behaviors": n_behaviors,
            "num_workers": num_workers,
            "adherence_threshold": adherence_threshold,
            "n_succeeded": sum(successes),
            "n_failed": n_behaviors - sum(successes),
            "n_attempts": n_attempts,
            "avg_latencies": {
                "generation": sum(a["latencies"]["generation"] for a in attempt_summaries) / n_attempts if n_attempts else 0,
                "rollout": sum(a["latencies"]["rollout"] for a in attempt_summaries) / n_attempts if n_attempts else 0,
                "evaluation": sum(a["latencies"]["evaluation"] for a in attempt_summaries) / n_attempts if n_attempts else 0,
            },
            "total_time": total_time,
            "num_attempts_to_success": {i: attempts_to_success[i] for i in range(n_behaviors)},
            "attempt_summaries": attempt_summaries,
        }

        # Save overall summary
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(overall_summary, f, indent=2)

        logging.info(f"Pipeline complete: {overall_summary['n_succeeded']}/{n_behaviors} behaviors succeeded in {n_attempts} attempts (Time: {overall_summary['total_time']:.1f}s)")

        return successes, overall_summary

    @track_time("Rollouts")
    def _run_rollouts(
        self,
        behavior_dirs: list[str],
        loading_kwargs_list: list[dict],
        n_rollouts: int,
        num_workers: int,
        **rollout_kwargs,
    ) -> list[list[dict]]:
        """Run rollouts with generated policy."""
        rollout_args = []
        for behavior_idx, (behavior_dir, loading_kwargs) in enumerate(zip(behavior_dirs, loading_kwargs_list)):
            output_path = f"{behavior_dir}/rollouts"
            os.makedirs(output_path, exist_ok=True)

            for rollout_idx in range(n_rollouts):
                rollout_args.append(
                    {
                        "rollout_fn": self.rollout_fn,
                        "policy_type": self.policy_type,
                        "loading_kwargs": loading_kwargs,
                        "rollout_subfolder": f"{output_path}/rollout_{rollout_idx}",
                        "rollout_kwargs": rollout_kwargs,
                        "_behavior_idx": behavior_idx,
                        "_rollout_idx": rollout_idx,
                    }
                )

        logging.info(f"Running {len(rollout_args)} rollouts ({len(behavior_dirs)} behaviors, {n_rollouts} each) with {num_workers} workers...")
        rollout_results = pqdm(
            rollout_args,
            _worker_rollout,
            n_jobs=num_workers,
            desc="Running rollouts",
        )

        n_behaviors = len(behavior_dirs)
        behavior_rollouts: list[list[dict]] = [[] for _ in range(n_behaviors)]

        for args, result in zip(rollout_args, rollout_results):
            behavior_idx = args["_behavior_idx"]
            behavior_rollouts[behavior_idx].append(result)

        return behavior_rollouts

    @track_time("Policy evaluation")
    def _evaluate_policies(
        self,
        behaviors: list[Behavior],
        behaviors_rollouts: list[list[dict]],
        behavior_dirs: list[str],
        environment_description: str,
        **kwargs,
    ) -> Any:
        """Evaluate policies for a batch of behaviors using parallel rollouts.

        Returns:
            EvalResult object with evaluation results
        """
        # Build evaluator inputs
        eval_inputs = []
        for behavior_idx, behavior in enumerate(behaviors):
            rollouts = behaviors_rollouts[behavior_idx]

            rollout_states = []
            rollout_images = []
            for result in rollouts:
                if isinstance(result, Exception):
                    rollout_states.append(f"[ERROR] Rollout: {result}")
                    rollout_images.append(None)
                elif result["error"] is not None:
                    rollout_states.append(f"[ERROR] Rollout: {result['error']}")
                    rollout_images.append(None)
                else:
                    rollout_states.append(result["states"])
                    rollout_images.append(result["images"])

            eval_inputs.append(
                {
                    "behavior": str(behavior),
                    "environment_description": environment_description,
                    "rollout_states": rollout_states,
                    "rollout_images": rollout_images,
                }
            )

        # Run evaluation
        logging.info(f"Evaluating {len(eval_inputs)} behaviors...")
        eval_result = self.evaluator.evaluate(inputs=eval_inputs)

        # Save per-behavior results
        for idx, behavior_dir in enumerate(behavior_dirs):
            instance_result = {
                "instance_metrics": [eval_result.instance_metrics[idx]],
                "overall_metrics": eval_result.instance_metrics[idx],
                "metadata": eval_result.metadata,
            }
            with open(f"{behavior_dir}/evaluation.json", "w") as f:
                json.dump(instance_result, f, indent=2)

        return eval_result


class PolicyCodeAgentGenerationPipeline(AgentGenerationPipeline):
    """Pipeline for generating policies with LLM feedback."""

    policy_type = "PolicyCode"

    @track_time("Policy generation")
    def _generate_policies(
        self,
        behaviors: list[Behavior],
        environment_description: str,
        feedbacks: list[list[tuple[str, str]]],
        output_dirs: list[str],
        num_workers: int = 1,
        **kwargs,
    ) -> tuple[list[Any], list[dict]]:
        """Generate policies using LLM with feedback."""
        loading_kwargs = []
        generated_code_policies = []

        logging.info(f"Generating {len(behaviors)} policies with {num_workers} workers...")
        inputs = [
            {
                "behavior": str(b),
                "feedbacks": fb,
                "environment_description": environment_description,
                "examples": None,  # TODO: add examples to generation kwargs if needed
            }
            for b, fb in zip(behaviors, feedbacks)
        ]
        generated_codes = self.generator.batch_generate(
            inputs=inputs,
            num_workers=num_workers,
            **kwargs,
        )

        for output_dir, gen_code in zip(output_dirs, generated_codes):
            code_file_path = f"{output_dir}/policy.py"
            loading_kwargs.append({"file_path": code_file_path})
            generated_code_policies.append(gen_code)

            # Save generated policy code
            with open(code_file_path, "w") as f:
                f.write(gen_code.code)
            logging.debug(f"Saved generated policy to {code_file_path}")

        return generated_code_policies, loading_kwargs


class RLAgentGenerationPipeline(AgentGenerationPipeline):
    """Pipeline for generating reward functions and training RL policies."""

    policy_type = "RewardFn"

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BehaviorAdherenceEvaluator,
        rollout_fn: callable,
        output_dir: str,
        train_fn: callable,
        training_attempts: int = 1,
    ):
        super().__init__(
            generator=generator,
            evaluator=evaluator,
            rollout_fn=rollout_fn,
            output_dir=output_dir,
        )
        self.train_fn = train_fn
        self.training_attempts = training_attempts

    @track_time("Reward generation and RL training")
    def _generate_policies(
        self,
        behaviors: list[Behavior],
        environment_description: str,
        examples: list[str],
        feedbacks: list[list[tuple[str, str]]],
        output_dirs: list[str],
        num_workers: int = 1,
        **kwargs,
    ) -> tuple[list[Any], list[dict]]:
        """Generate reward functions and train RL policies."""
        # Batch generate reward functions
        logging.info(f"Generating {len(behaviors)} reward functions with {num_workers} workers...")
        inputs = [
            {
                "behavior": str(b),
                "feedbacks": fb,
                "environment_description": environment_description,
                "examples": examples,
            }
            for b, fb in zip(behaviors, feedbacks)
        ]
        generated_codes = self.generator.batch_generate(
            inputs=inputs,
            num_workers=num_workers,
            **kwargs,
        )

        # Save rewards and train RL policies
        generated_policies = []
        loading_kwargs = []
        for i, (output_dir, gen_code) in enumerate(zip(output_dirs, generated_codes)):
            code_file_path = f"{output_dir}/reward.py"
            loading_kwargs.append({"save_dir": output_dir})

            # Save generated reward
            with open(code_file_path, "w") as f:
                f.write(gen_code.code)
            logging.debug(f"Saved generated reward to {code_file_path}")

            # Train RL policy with retries
            logging.info(f"[{i}] Training RL policy...")
            compute_reward = import_function_from_file(code_file_path, "compute_reward")
            for attempt in range(self.training_attempts):
                try:
                    self.train_fn(compute_reward, output_dir)
                    logging.info(f"[{i}] Training complete.")
                    break
                except Exception as e:
                    if attempt < self.training_attempts - 1:
                        logging.warning(f"[{i}] Training failed (attempt {attempt + 1}/{self.training_attempts}): {e}. Retrying...")
                    else:
                        logging.error(f"[{i}] Training failed after {self.training_attempts} attempts: {e}")
                        gen_code = e

            generated_policies.append(gen_code)

        return generated_policies, loading_kwargs


class LLMAgentGenerationPipeline(AgentGenerationPipeline):
    """Pipeline for generating LLM-based agents with behavior specifications."""

    policy_type = "LLMAgent"

    def __init__(
        self,
        agent_model_config: LitellmConfig,
        evaluator: BehaviorAdherenceEvaluator,
        rollout_fn: callable,
        output_dir: str,
    ):
        super().__init__(
            generator=None,  # Not used in this pipeline
            evaluator=evaluator,
            rollout_fn=rollout_fn,
            output_dir=output_dir,
        )
        self.agent_model_config = agent_model_config

    @track_time("LLM Agent configuration")
    def _generate_policies(
        self,
        behaviors: list[Behavior],
        output_dirs: list[str],
        environment_description: str = "",
        action_type: Literal["Discrete", "Continuous"] = None,
        horizon_length: int = 1,
        include_trajectory_history: bool = False,
        **kwargs,
    ) -> tuple[list[Any], list[dict]]:
        """Generate LLM agent configurations for each behavior."""
        loading_kwargs = []
        generated_configs = []

        logging.info(f"Creating LLM agent configurations for {len(behaviors)} behaviors...")

        for output_dir, behavior in zip(output_dirs, behaviors):
            # Create configuration for this LLM agent
            config = {
                "behavior": behavior.description,
                "model_config": {
                    "model_name": self.agent_model_config.model_name,
                    "api_base": self.agent_model_config.api_base,
                    "generation_kwargs": self.agent_model_config.generation_kwargs,
                },
                "environment_description": environment_description,
                "action_type_module": f"{action_type.__module__}:{action_type.__qualname__}",
                "horizon_length": horizon_length,
                "include_trajectory_history": include_trajectory_history,
            }

            # Save configuration to JSON
            config_file_path = f"{output_dir}/llm_agent_config.json"
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=2)

            # Add loading kwargs for this agent
            loading_kwargs.append({"config_path": config_file_path})
            generated_configs.append(behavior.description)

        return generated_configs, loading_kwargs
