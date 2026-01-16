import json
import logging
import os
from typing import Any

from pqdm.processes import pqdm

from popgen.utils.decorators import track_time
from popgen.utils.imports import import_all_functions_from_file
from popgen.utils.logging import save_rollout, uniform_sample

from .evaluation import BehaviorAdherenceEvaluator
from .generation import BaseGenerator, PolicyCode
from .types import Behavior

logging.basicConfig(level=logging.INFO)


def _worker_rollout(args: dict) -> dict:
    """Worker function for parallel rollout execution."""
    rollout_fn = args["rollout_fn"]
    policies_fns = import_all_functions_from_file(file_path=args["policy_code_path"])
    rollout_subfolder = args["rollout_subfolder"]
    kwargs = args["kwargs"]

    # Perform rollout
    rollout_data, images, error = rollout_fn(**policies_fns, **kwargs)

    # Save rollout trajectory and images
    save_rollout(
        rollout_data=rollout_data,
        images=images,
        save_path=rollout_subfolder,
    )

    if error is not None:
        return {"error": str(error), "states": None, "images": None}

    return {
        "error": None,
        "states": uniform_sample(rollout_data, n_samples=20),
        "images": uniform_sample(images, n_samples=5),
    }


class FeedbackGenerationPipeline:
    """Pipeline for generating diverse behaviors, policies, and rewards."""

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BehaviorAdherenceEvaluator,
        rollout_fn: callable,
        output_dir: str,
    ):
        self.generator: BaseGenerator = generator
        self.evaluator: BehaviorAdherenceEvaluator = evaluator
        self.output_dir = output_dir
        self.rollout_fn = rollout_fn
        self.item_type = self.generator.generation_type

        os.makedirs(self.output_dir, exist_ok=True)

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
        """Run the feedback generation pipeline for a list of behaviors.

        Args:
            behaviors: List of Behavior objects
            num_attempts: Maximum number of alignment attempts per behavior
            n_rollouts: Number of rollouts per behavior for evaluation
            num_workers: Number of parallel workers for generation and rollouts
            generation_kwargs: Additional kwargs for generation
            evaluation_kwargs: Additional kwargs for evaluation

        Returns:
            Tuple of (success_flags, overall_summary)
        """
        n_behaviors = len(behaviors)

        # Track state per behavior
        feedbacks: list[list[tuple[str, str]]] = [[] for _ in range(n_behaviors)]
        policy_code_paths: list[str] = [None] * n_behaviors
        generated_items: list[PolicyCode] = [None] * n_behaviors
        successes: list[bool] = [False] * n_behaviors
        steps_to_success: list[int] = [None] * n_behaviors  # None means not succeeded

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
            pending_feedbacks = [feedbacks[i][-1:] for i in pending_indices]  # only latest feedback

            generation_results, generation_latency = self._generate_policies(
                behaviors=pending_behaviors,
                feedbacks=pending_feedbacks,
                behavior_indices=pending_indices,
                attempt=attempt,
                num_workers=num_workers,
                **generation_kwargs,
            )

            # Update policy paths and generated items for pending behaviors
            for idx, (code_path, gen_item) in zip(pending_indices, generation_results):
                policy_code_paths[idx] = code_path
                generated_items[idx] = gen_item

            # Step 2: Run rollouts in parallel for all pending behaviors
            eval_results, evaluation_latency = self._evaluate_policies(
                behaviors=pending_behaviors,
                behavior_indices=pending_indices,
                policy_code_paths=[policy_code_paths[i] for i in pending_indices],
                attempt=attempt,
                n_rollouts=n_rollouts,
                num_workers=num_workers,
                **evaluation_kwargs,
            )

            # Step 3: Process results and update state
            behavior_results = {}
            for i, idx in enumerate(pending_indices):
                instance_metric = eval_results.instance_metrics[i]
                adherence = instance_metric["adherence"]
                success = adherence >= adherence_threshold
                successes[idx] = success
                behavior_results[idx] = success

                if success:
                    steps_to_success[idx] = attempt + 1
                    logging.info(f"[B={idx}] SUCCESS on attempt {attempt + 1}")
                else:
                    logging.warning(f"[B={idx}] FAILURE – Adherence: {adherence:.2f}")
                    overall_feedback = instance_metric["reasoning"]
                    feedbacks[idx].append((generated_items[idx].code, overall_feedback))

            # Log per-attempt summary
            attempt_summary = {
                "attempt": attempt + 1,
                "latencies": {
                    "generation": generation_latency,
                    "evaluation": evaluation_latency,
                    "total": generation_latency + evaluation_latency,
                },
                "behavior_results": behavior_results,
                "n_succeeded": sum(behavior_results.values()),
                "n_pending": len(pending_indices),
            }
            attempt_summaries.append(attempt_summary)
            logging.info(f"[Attempt {attempt + 1}] Gen: {generation_latency:.1f}s, Eval: {evaluation_latency:.1f}s, Success: {attempt_summary['n_succeeded']}/{len(pending_indices)}")

        # Compute overall summary
        total_generation_latency = sum(a["latencies"]["generation"] for a in attempt_summaries)
        total_evaluation_latency = sum(a["latencies"]["evaluation"] for a in attempt_summaries)
        n_attempts = len(attempt_summaries)

        overall_summary = {
            "n_behaviors": n_behaviors,
            "n_succeeded": sum(successes),
            "n_attempts": n_attempts,
            "avg_latencies": {
                "generation": total_generation_latency / n_attempts if n_attempts else 0,
                "evaluation": total_evaluation_latency / n_attempts if n_attempts else 0,
            },
            "total_latencies": {
                "generation": total_generation_latency,
                "evaluation": total_evaluation_latency,
            },
            "behavior_steps_to_success": {i: steps_to_success[i] for i in range(n_behaviors)},
            "attempt_summaries": attempt_summaries,
        }

        # Save overall summary
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(overall_summary, f, indent=2)

        logging.info(
            f"Pipeline complete: {overall_summary['n_succeeded']}/{n_behaviors} behaviors succeeded in {n_attempts} attempts (Total: {total_generation_latency + total_evaluation_latency:.1f}s)"
        )

        return successes, overall_summary

    @track_time("Policy generation")
    def _generate_policies(
        self,
        behaviors: list[Behavior],
        feedbacks: list[list[tuple[str, str]]],
        behavior_indices: list[int],
        attempt: int,
        num_workers: int = 1,
        **kwargs,
    ) -> list[tuple[str, PolicyCode]]:
        """Generate policies for a batch of behaviors using parallel generation.

        Returns:
            List of (code_path, generated_item) tuples
        """
        # Prepare output paths and check cache
        code_file_paths = []
        cached_results = {}
        behaviors_to_generate = []

        for behavior, fb, idx in zip(behaviors, feedbacks, behavior_indices):
            attempt_output_dir = os.path.join(self.output_dir, f"behavior_{idx}", f"attempt_{attempt}")
            os.makedirs(attempt_output_dir, exist_ok=True)
            code_file_path = f"{attempt_output_dir}/{self.item_type.lower()}.py"
            code_file_paths.append(code_file_path)

            if os.path.exists(code_file_path):
                logging.info(f"[B={idx}] Loading cached {self.item_type} from {code_file_path}")
                with open(code_file_path) as f:
                    code = f.read()
                cached_results[idx] = PolicyCode(code=code, reasoning=None)
            else:
                behaviors_to_generate.append(
                    {
                        "behavior": behavior,
                        "feedbacks": fb,
                        "idx": idx,
                        "code_file_path": code_file_path,
                    }
                )

        # Batch generate non-cached behaviors
        if behaviors_to_generate:
            logging.info(f"Generating {len(behaviors_to_generate)} policies with {num_workers} workers...")
            generated_codes = self.generator.batch_generate(
                behaviors=[str(b["behavior"]) for b in behaviors_to_generate],
                feedbacks_list=[b["feedbacks"] for b in behaviors_to_generate],
                num_workers=num_workers,
                **kwargs,
            )

            # Save generated code and update results
            for item, gen_code in zip(behaviors_to_generate, generated_codes):
                with open(item["code_file_path"], "w") as f:
                    f.write(gen_code.code)
                logging.debug(f"[B={item['idx']}] Saved generated {self.item_type}")
                cached_results[item["idx"]] = gen_code

        # Reconstruct results in original order
        results = []
        for idx, code_file_path in zip(behavior_indices, code_file_paths):
            results.append((code_file_path, cached_results[idx]))

        return results

    @track_time("Policy evaluation")
    def _evaluate_policies(
        self,
        behaviors: list[Behavior],
        behavior_indices: list[int],
        policy_code_paths: list[str],
        attempt: int,
        n_rollouts: int,
        num_workers: int,
        ego_name: str,
        ego_visual_description: str,
        **kwargs,
    ) -> list[Any]:
        """Evaluate policies for a batch of behaviors using parallel rollouts.

        Returns:
            List of EvalResult objects, one per behavior
        """
        # Prepare rollout arguments for all (behavior, rollout_idx) combinations
        rollout_args = []
        for behavior_idx, (idx, code_path) in enumerate(zip(behavior_indices, policy_code_paths)):
            attempt_output_dir = os.path.join(self.output_dir, f"behavior_{idx}", f"attempt_{attempt}")
            rollouts_output_dir = f"{attempt_output_dir}/rollouts"
            os.makedirs(rollouts_output_dir, exist_ok=True)

            for rollout_idx in range(n_rollouts):
                rollout_args.append(
                    {
                        "rollout_fn": self.rollout_fn,
                        "policy_code_path": code_path,
                        "rollout_subfolder": f"{rollouts_output_dir}/rollout_{rollout_idx}",
                        "kwargs": kwargs,
                        # Metadata for reconstruction
                        "_behavior_idx": behavior_idx,
                        "_rollout_idx": rollout_idx,
                    }
                )

        # Run all rollouts in parallel
        logging.info(f"Running {len(rollout_args)} rollouts ({len(behaviors)} behaviors, {n_rollouts} rollouts each) with {num_workers} workers...")
        rollout_results = pqdm(
            rollout_args,
            _worker_rollout,
            n_jobs=num_workers,
            desc="Running rollouts",
        )

        # Reconstruct results per behavior
        n_behaviors = len(behaviors)
        behavior_rollouts: list[list[dict]] = [[] for _ in range(n_behaviors)]

        for args, result in zip(rollout_args, rollout_results):
            behavior_idx = args["_behavior_idx"]
            behavior_rollouts[behavior_idx].append(result)

        # Build all evaluator inputs
        eval_inputs = []
        for behavior_idx, behavior in enumerate(behaviors):
            rollouts = behavior_rollouts[behavior_idx]

            # Collect rollout states and images
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
                    "ego_name": ego_name,
                    "ego_visual_description": ego_visual_description,
                    "rollout_states": rollout_states,
                    "rollout_images": rollout_images,
                }
            )

        # Run evaluation on all behaviors in parallel
        logging.info(f"Evaluating {len(eval_inputs)} behaviors...")
        eval_result = self.evaluator.evaluate(inputs=eval_inputs)

        # Save per-behavior results
        for behavior_idx, idx in enumerate(behavior_indices):
            attempt_output_dir = os.path.join(self.output_dir, f"behavior_{idx}", f"attempt_{attempt}")
            # Save individual instance metric
            instance_result = {
                "instance_metrics": [eval_result.instance_metrics[behavior_idx]],
                "overall_metrics": eval_result.instance_metrics[behavior_idx],
                "metadata": eval_result.metadata,
            }
            with open(f"{attempt_output_dir}/evaluation.json", "w") as f:
                json.dump(instance_result, f, indent=2)

        return eval_result
