import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any, Literal

from pqdm.processes import pqdm

from popgen.agents import PolicyCodeAgent
from popgen.population.evaluation.utils import combine_feedback
from popgen.utils.config import LitellmConfig
from popgen.utils.decorators import track_time
from popgen.utils.imports import import_all_functions_from_file, import_function_from_file, load_entrypoint
from popgen.utils.logging import save_rollout, uniform_sample

from .evaluation import BehaviorAdherenceEvaluator
from .generation import BaseGenerator
from .types import Behavior

logging.basicConfig(level=logging.INFO)


def load_policy_and_env(policy_type: Literal["PolicyCode", "RewardFn", "LLMAgent", "Trajectory"], create_env_fn: callable, scripted_policy_fn: callable = None, **kwargs) -> dict:
    """Load policy functions and environment.

    Returns:
        Dict with:
            - env: environment instance
            - policies: list of policies passed into rollout_fn
              (always [ego_policy], or [ego_policy, teammate_policy] when scripted_policy_fn is provided)
    """
    if policy_type == "PolicyCode":
        policies = import_all_functions_from_file(kwargs["file_path"])
        policies = {name: PolicyCodeAgent(func) for name, func in policies.items()}
        if "policy" not in policies:
            raise KeyError("Expected generated policy file to define a callable named 'policy'.")
        ego_policy = policies["policy"]
        env = create_env_fn(num_envs=1)
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
        # Group key is "ego" when trained with a scripted teammate, "agents" otherwise.
        ego_group = "ego" if "ego" in experiment.group_policies else "agents"
        ego_policy = experiment.group_policies[ego_group]
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
            feedbacks=config.get("feedbacks", []),
        )

        ego_policy = policy
        env = create_env_fn(num_envs=1)
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    policies = [ego_policy]
    if scripted_policy_fn is not None:
        teammate_policy = scripted_policy_fn(env)
        policies.append(teammate_policy)

    return {"env": env, "policies": policies}


# ---------------------------------- WORKERS --------------------------------- #


@track_time("rollout")
def rollout_worker(args: dict) -> dict:
    """Worker function for parallel rollout execution."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Import environment module
    rollout_fn = args["rollout_fn"]
    artifacts_dict = load_policy_and_env(policy_type=args["policy_type"], create_env_fn=args["create_env_fn"], scripted_policy_fn=args["scripted_policy_fn"], **args.get("loading_kwargs", {}))

    # Perform rollout
    rollout_data, images, error = rollout_fn(artifacts_dict["policies"], env=artifacts_dict["env"], **args["rollout_kwargs"])

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


@track_time("training")
def train_worker(args: dict) -> dict:
    """Worker function for parallel RL training."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    compute_reward = import_function_from_file(args["code_file_path"], "compute_reward")
    for attempt in range(args["training_attempts"]):
        try:
            args["train_fn"](compute_reward, args["output_dir"])
            logging.info(f"[{args['idx']}] Training complete.")
            return {"error": None}
        except Exception as e:
            if attempt < args["training_attempts"] - 1:
                logging.warning(f"[{args['idx']}] Training failed (attempt {attempt + 1}/{args['training_attempts']}): {e}. Retrying...")
            else:
                logging.error(f"[{args['idx']}] Training failed after {args['training_attempts']} attempts: {e}")
                return {"error": e}
    return {"error": None}


# ---------------------------------------------------------------------------- #


class AgentGenerationPipeline(ABC):
    """Base pipeline for behavior-aligned policy generation and evaluation."""

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BehaviorAdherenceEvaluator,
        env_module: Any,
        output_dir: str,
    ):
        self.generator: BaseGenerator = generator
        self.evaluator: BehaviorAdherenceEvaluator = evaluator
        self.env_module = env_module
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def _generate_policies(
        self,
        *args,
        **kwargs,
    ) -> tuple[list[Any], list[dict], list[float]]:
        """Generate policies for a batch of behaviors.

        Returns:
            Tuple of (generated_policies, loading_kwargs_list, gen_latencies)
            where gen_latencies[i] is the per-behavior generation wall-clock time.
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
            pending_feedbacks = [feedbacks[i][-1:] for i in pending_indices]  # Only pass the most recent feedback for generation to avoid overwhelming the model with long feedback histories.

            pending_output_dirs = []
            for i in pending_indices:
                dir = os.path.join(self.output_dir, f"behavior_{i}", f"attempt_{attempt}")
                pending_output_dirs.append(dir)
                os.makedirs(dir, exist_ok=True)

            t0 = time()
            gen_results, loading_policy_kwargs, per_gen_latencies = self._generate_policies(
                behaviors=pending_behaviors,
                feedbacks=pending_feedbacks,
                output_dirs=pending_output_dirs,
                num_workers=num_workers,
                **generation_kwargs,
            )
            gen_latency = time() - t0

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
                t0 = time()
                behaviors_rollouts, per_rollout_latencies = self._run_rollouts(
                    n_rollouts=n_rollouts,
                    attempt=attempt,
                    num_workers=num_workers,
                    loading_kwargs_list=valid_loading_kwargs,
                    behavior_dirs=valid_behavior_dirs,
                    **evaluation_kwargs,
                )
                rollout_latency = time() - t0

                t0 = time()
                eval_results, per_eval_latencies = self._evaluate_policies(
                    behaviors=valid_behaviors,
                    behaviors_rollouts=behaviors_rollouts,
                    behavior_dirs=valid_behavior_dirs,
                    num_workers=num_workers,
                    **evaluation_kwargs,
                )
                evaluation_latency = time() - t0

                # Process evaluation results for valid indices
                for i, idx in enumerate(valid_indices):
                    behavior_eval_results = eval_results[i]
                    adherence = behavior_eval_results.overall_metrics["adherence"]
                    success = adherence >= adherence_threshold

                    # Map results back to original behavior index
                    successes[idx] = success

                    if success:
                        attempts_to_success[idx] = attempt + 1
                        logging.info(f"[B={idx}] SUCCESS on attempt {attempt + 1}")
                    else:
                        logging.warning(f"[B={idx}] FAILURE – Adherence: {adherence:.2f}")
                        overall_feedback = combine_feedback(behavior_eval_results.instance_metrics)
                        # For code-based pipelines, include the generated code in feedback.
                        # For other pipelines (e.g., LLM agents), include the full trajectory.
                        if hasattr(generated_items[idx], "code"):
                            feedbacks[idx].append((generated_items[idx].code, overall_feedback))
                        else:
                            feedbacks[idx].append(("", overall_feedback))
            else:
                rollout_latency = 0.0
                evaluation_latency = 0.0
                per_rollout_latencies = []
                per_eval_latencies = []

            # Save per-behavior latency.json for this attempt
            valid_rollout_lat = {idx: per_rollout_latencies[j] for j, idx in enumerate(valid_indices)}
            valid_eval_lat = {idx: per_eval_latencies[j] for j, idx in enumerate(valid_indices)}
            for local_i, (global_idx, output_dir) in enumerate(zip(pending_indices, pending_output_dirs)):
                latency_record = {
                    "generation": per_gen_latencies[local_i],
                    "rollout": valid_rollout_lat.get(global_idx, []),
                    "evaluation": valid_eval_lat.get(global_idx, 0.0),
                }
                with open(os.path.join(output_dir, "latency.json"), "w") as f:
                    json.dump(latency_record, f, indent=2)

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

    def _run_rollouts(
        self,
        behavior_dirs: list[str],
        loading_kwargs_list: list[dict],
        n_rollouts: int,
        num_workers: int,
        seed: int = 0,
        **rollout_kwargs,
    ) -> list[list[dict]]:
        """Run rollouts with generated policy."""
        rollout_args = []
        for behavior_idx, (behavior_dir, loading_kwargs) in enumerate(zip(behavior_dirs, loading_kwargs_list)):
            output_path = f"{behavior_dir}/rollouts"
            os.makedirs(output_path, exist_ok=True)

            for rollout_idx in range(n_rollouts):
                rollout_seed = seed + rollout_idx * 1000
                rollout_args.append(
                    {
                        "rollout_fn": self.env_module.rollout_fn,
                        "policy_type": self.policy_type,
                        "create_env_fn": self.env_module.create_env,
                        "scripted_policy_fn": self.env_module.ScriptedPolicy,
                        "loading_kwargs": loading_kwargs,
                        "rollout_subfolder": f"{output_path}/rollout_{rollout_idx}",
                        "rollout_kwargs": {**rollout_kwargs, "seed": rollout_seed},
                        "_behavior_idx": behavior_idx,
                        "_rollout_idx": rollout_idx,
                    }
                )

        logging.info(f"Running {len(rollout_args)} rollouts ({len(behavior_dirs)} behaviors, {n_rollouts} each) with {num_workers} workers...")
        rollout_results = pqdm(
            rollout_args,
            rollout_worker,
            n_jobs=num_workers,
            desc="Running rollouts",
        )

        n_behaviors = len(behavior_dirs)
        behavior_rollouts: list[list[dict]] = [[] for _ in range(n_behaviors)]
        behavior_latencies: list[list[float]] = [[] for _ in range(n_behaviors)]

        for args, item in zip(rollout_args, rollout_results):
            behavior_idx = args["_behavior_idx"]
            if isinstance(item, Exception):
                behavior_rollouts[behavior_idx].append(item)
                behavior_latencies[behavior_idx].append(0.0)
            else:
                result, elapsed = item
                behavior_rollouts[behavior_idx].append(result)
                behavior_latencies[behavior_idx].append(elapsed)

        return behavior_rollouts, behavior_latencies

    def _evaluate_policies(
        self,
        behaviors: list[Behavior],
        behaviors_rollouts: list[list[dict]],
        behavior_dirs: list[str],
        num_workers: int = 8,
        **kwargs,
    ) -> Any:
        """Evaluate policies for a batch of behaviors using parallel rollouts.

        Returns:
            EvalResult object with evaluation results
        """
        environment_description = self.env_module.ENVIRONMENT_DESCRIPTION

        # Build evaluator inputs
        all_eval_inputs = []
        for behavior_idx, behavior in enumerate(behaviors):
            behavior_eval_inputs = []
            rollouts = behaviors_rollouts[behavior_idx]

            rollout_states = []
            rollout_images = []
            for result in rollouts:
                if isinstance(result, Exception):
                    rollout_states = f"[ERROR] Rollout: {result}"
                    rollout_images = None
                elif result["error"] is not None:
                    rollout_states = f"[ERROR] Rollout: {result['error']}"
                    rollout_images = None
                else:
                    rollout_states = result["states"]
                    rollout_images = result["images"]

                behavior_eval_inputs.append(
                    {
                        "behavior": str(behavior),
                        "environment_description": environment_description,
                        "rollout_states": rollout_states,
                        "rollout_images": rollout_images,
                    }
                )
            all_eval_inputs.append(behavior_eval_inputs)

        # Run evaluation with per-behavior timing
        logging.info(f"Evaluating {len(all_eval_inputs)} behaviors...")
        num_parallel_eval = max(1, num_workers // self.evaluator.num_workers)

        timed_evaluate = track_time("evaluate")(self.evaluator.evaluate)
        timed_eval = pqdm(all_eval_inputs, timed_evaluate, n_jobs=num_parallel_eval, desc="Running evaluations")
        eval_result, eval_latencies = [], []
        for item in timed_eval:
            if isinstance(item, Exception):
                eval_result.append(item)
                eval_latencies.append(0.0)
            else:
                eval_result.append(item[0])
                eval_latencies.append(item[1])

        # Add rollout data paths to instance metrics
        for idx, behavior_dir in enumerate(behavior_dirs):
            for rollout_idx, instance_metric in enumerate(eval_result[idx].instance_metrics):
                instance_metric["rollout_data_path"] = os.path.abspath(f"{behavior_dir}/rollouts/rollout_{rollout_idx}/rollout_data.jsonl")

        # Save per-behavior results
        for idx, behavior_dir in enumerate(behavior_dirs):
            instance_result = {
                "instance_metrics": eval_result[idx].instance_metrics,
                "overall_metrics": eval_result[idx].overall_metrics,
                "metadata": eval_result[idx].metadata,
            }
            with open(f"{behavior_dir}/evaluation.json", "w") as f:
                json.dump(instance_result, f, indent=2)

        return eval_result, eval_latencies


class PolicyCodeAgentGenerationPipeline(AgentGenerationPipeline):
    """Pipeline for generating policies with LLM feedback."""

    policy_type = "PolicyCode"

    def _generate_policies(
        self,
        behaviors: list[Behavior],
        feedbacks: list[list[tuple[str, str]]],
        output_dirs: list[str],
        num_workers: int = 1,
        **kwargs,
    ) -> tuple[list[Any], list[dict]]:
        """Generate policies using LLM with feedback."""
        loading_kwargs = []
        generated_code_policies = []

        environment_description = self.env_module.ENVIRONMENT_DESCRIPTION

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
        generated_codes, gen_latencies = self.generator.batch_generate(
            inputs=inputs,
            num_workers=num_workers,
            timed=True,
            **kwargs,
        )

        for output_dir, gen_code in zip(output_dirs, generated_codes):
            code_file_path = f"{output_dir}/policy.py"
            loading_kwargs.append({"file_path": code_file_path})
            generated_code_policies.append(gen_code)

            # Save generated policy code (or error if generation failed)
            with open(code_file_path, "w") as f:
                if isinstance(gen_code, Exception):
                    f.write(f"# Generation error: {gen_code}\n")
                else:
                    f.write(gen_code.code)
            logging.debug(f"Saved generated policy to {code_file_path}")

            # Save full model output to the attempt folder
            record = {"output_dir": output_dir}
            record.update(gen_code.model_dump() if not isinstance(gen_code, Exception) else {"error": str(gen_code)})
            with open(os.path.join(output_dir, "model_outputs.jsonl"), "w") as f:
                f.write(json.dumps(record) + "\n")

        return generated_code_policies, loading_kwargs, gen_latencies


class RLAgentGenerationPipeline(AgentGenerationPipeline):
    """Pipeline for generating reward functions and training RL policies."""

    policy_type = "RewardFn"

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BehaviorAdherenceEvaluator,
        env_module: Any,
        output_dir: str,
        training_attempts: int = 1,
    ):
        super().__init__(
            generator=generator,
            evaluator=evaluator,
            env_module=env_module,
            output_dir=output_dir,
        )
        self.training_attempts = training_attempts

    def _generate_policies(
        self,
        behaviors: list[Behavior],
        feedbacks: list[list[tuple[str, str]]],
        output_dirs: list[str],
        num_workers: int = 1,
        **kwargs,
    ) -> tuple[list[Any], list[dict], list[dict]]:
        """Generate reward functions and train RL policies."""
        environment_description = self.env_module.ENVIRONMENT_DESCRIPTION
        examples = [self.env_module.REWARD_EXAMPLE]

        # Step 1: Batch generate reward functions
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
        generated_codes, reward_gen_latencies = self.generator.batch_generate(
            inputs=inputs,
            num_workers=num_workers,
            timed=True,
            **kwargs,
        )

        # Step 2: Save reward files and prepare training args
        loading_kwargs = []
        train_args = []
        for i, (output_dir, gen_code) in enumerate(zip(output_dirs, generated_codes)):
            code_file_path = f"{output_dir}/reward.py"
            loading_kwargs.append({"save_dir": output_dir})

            with open(code_file_path, "w") as f:
                if isinstance(gen_code, Exception):
                    f.write(f"# Generation error: {gen_code}\n")
                else:
                    f.write(gen_code.code)
            logging.debug(f"Saved generated reward to {code_file_path}")

            record = {"output_dir": output_dir}
            record.update(gen_code.model_dump() if not isinstance(gen_code, Exception) else {"error": str(gen_code)})
            with open(os.path.join(output_dir, "model_outputs.jsonl"), "w") as f:
                f.write(json.dumps(record) + "\n")

            train_args.append(
                {
                    "code_file_path": code_file_path,
                    "output_dir": output_dir,
                    "train_fn": self.env_module.train_fn,
                    "training_attempts": self.training_attempts,
                    "idx": i,
                }
            )

        # Step 3: Train RL policies in parallel
        logging.info(f"Training {len(train_args)} RL policies with {num_workers} workers...")
        train_results = pqdm(
            train_args,
            train_worker,
            n_jobs=num_workers,
            desc="Training RL policies",
        )

        # Step 4: Collect results and per-behavior latencies
        generated_policies = []
        per_gen_latencies = []
        for i, (gen_code, result) in enumerate(zip(generated_codes, train_results)):
            if isinstance(result, Exception):
                generated_policies.append(result)
                train_latency = 0.0
            else:
                train_result, train_latency = result
                if train_result["error"] is not None:
                    generated_policies.append(train_result["error"])
                else:
                    generated_policies.append(gen_code)
            per_gen_latencies.append(
                {
                    "reward_generation": reward_gen_latencies[i],
                    "training": train_latency,
                }
            )

        return generated_policies, loading_kwargs, per_gen_latencies


class LLMAgentGenerationPipeline(AgentGenerationPipeline):
    """Pipeline for generating LLM-based agents with behavior specifications."""

    policy_type = "LLMAgent"

    def __init__(
        self,
        agent_model_config: LitellmConfig,
        evaluator: BehaviorAdherenceEvaluator,
        env_module: Any,
        output_dir: str,
    ):
        super().__init__(
            generator=None,  # Not used in this pipeline
            evaluator=evaluator,
            env_module=env_module,
            output_dir=output_dir,
        )
        self.agent_model_config = agent_model_config

    def _generate_policies(
        self,
        behaviors: list[Behavior],
        output_dirs: list[str],
        feedbacks: list[list[tuple[str, str]]] = None,
        horizon_length: int = 1,
        include_trajectory_history: bool = False,
    ) -> tuple[list[Any], list[dict]]:
        """Generate LLM agent configurations for each behavior."""
        loading_kwargs = []
        generated_configs = []

        environment_description = self.env_module.ENVIRONMENT_DESCRIPTION
        action_type = self.env_module.ActionType

        logging.info(f"Creating LLM agent configurations for {len(behaviors)} behaviors...")

        for bi, (output_dir, behavior) in enumerate(zip(output_dirs, behaviors)):
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
                "feedbacks": feedbacks[bi] if feedbacks else [],
            }

            # Save configuration to JSON
            config_file_path = f"{output_dir}/llm_agent_config.json"
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=2)

            # Save model outputs record — include config_path as attribute of the wrapper
            record = {"config_path": config_file_path}
            record.update(config)
            with open(os.path.join(output_dir, "model_outputs.jsonl"), "w") as f:
                f.write(json.dumps(record) + "\n")

            # Add loading kwargs for this agent
            loading_kwargs.append({"config_path": config_file_path})
            generated_configs.append(behavior.description)

        # No LLM generation step — generation latency is 0
        gen_latencies = [0.0] * len(behaviors)
        return generated_configs, loading_kwargs, gen_latencies
