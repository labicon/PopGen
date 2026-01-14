import json
import logging
import os
import time

from popgen.utils.imports import import_all_functions_from_file
from popgen.utils.logging import save_rollout, uniform_sample

from .evaluator import BehaviorEvaluator, EvaluationResult
from .generation import BaseGenerator

logging.basicConfig(level=logging.INFO)


class FeedbackGenerationPipeline:
    """Pipeline for generating diverse behaviors, policies, and rewards."""

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BehaviorEvaluator,
        rollout_fn: callable,
        output_dir: str,
    ):
        self.generator: BaseGenerator = generator
        self.evaluator: BehaviorEvaluator = evaluator
        self.output_dir = output_dir
        self.rollout_fn = rollout_fn
        self.item_type = self.generator.generation_type

        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(
        self,
        behavior: str,
        behavior_index: int,
        num_attempts: int = 5,
        generation_kwargs: dict = {},
        evaluation_kwargs: dict = {},
    ) -> dict:
        feedbacks = []

        # Initialize index for logging
        self.behavior_idx = behavior_index

        behavior_output_dir = os.path.join(self.output_dir, f"behavior_{self.behavior_idx}")

        for attempt in range(num_attempts):
            self.align_idx = attempt
            self.log_prefix = self._log_prefix()
            attempt_output_dir = os.path.join(behavior_output_dir, f"attempt_{attempt}")

            logging.info(f"{self.log_prefix} Generating policy...")
            generation_start = time.time()
            policies_fns, generated_item = self.generate_policies(
                behavior=behavior,
                feedbacks=feedbacks,
                generation_output_dir=attempt_output_dir,
                **generation_kwargs,
            )
            generation_latency = time.time() - generation_start

            rollouts_output_dir = f"{attempt_output_dir}/rollouts"
            os.makedirs(rollouts_output_dir, exist_ok=True)

            evaluation_start = time.time()
            try:
                evaluation_result = self.evaluate_policies(
                    behavior=behavior,
                    rollout_fn=self.rollout_fn,
                    policies_fns=policies_fns,
                    rollouts_output_dir=rollouts_output_dir,
                    **evaluation_kwargs,
                )
            except Exception as e:
                logging.error(f"{self.log_prefix} Error during rollout on attempt {attempt + 1}: {e}")
                evaluation_result = EvaluationResult(
                    evaluation=False,
                    reasoning=f"Rollout failed: {e}",
                )
            evaluation_latency = time.time() - evaluation_start

            output = {
                "behavior": behavior.model_dump(),
                "generation": {
                    "attempts": attempt + 1,
                    "code": generated_item.code,
                    "reasoning": getattr(generated_item, "reasoning", "N/A"),
                    "latency": generation_latency,
                },
                "evaluation": {
                    "rollouts_path": rollouts_output_dir,
                    "result": evaluation_result.model_dump(),
                    "latency": evaluation_latency,
                },
            }

            evaluation_output_path = f"{attempt_output_dir}/evaluation.json"
            with open(evaluation_output_path, "w") as f:
                json.dump(output, f, indent=2)

            if evaluation_result.evaluation:
                logging.info(f"{self.log_prefix} SUCCESS – Alignment succeeded on attempt {attempt + 1}")
                return True, output
            else:
                logging.warning(f"{self.log_prefix} FAILURE – Alignment failed on attempt {attempt + 1}. Reason: {evaluation_result.reasoning[:200]}[...]")
                feedbacks.append((generated_item.code, evaluation_result.reasoning))
                if attempt < num_attempts - 1:
                    logging.info(f"{self.log_prefix} Retrying alignment ({attempt + 2}/{num_attempts})...")

        logging.error(f"{self.log_prefix} All {num_attempts} attempts exhausted. Alignment failed.")
        return False, output

    def generate_policies(self, behavior: str, feedbacks: list, generation_output_dir: str, **kwargs):
        generated_code = self.generator.generate(behavior=behavior, feedbacks=feedbacks, **kwargs)

        logging.debug(f"{self.log_prefix} Generated {self.item_type}:\n{generated_code.code}")

        # save generated code to file
        os.makedirs(generation_output_dir, exist_ok=True)
        code_file_path = f"{generation_output_dir}/{self.item_type.lower()}.py"
        with open(code_file_path, "w") as f:
            f.write(generated_code.code)
        logging.debug(f"{self.log_prefix} Saved generated {self.item_type} to {code_file_path}")

        # Import policy functions from generated code
        policies_fns = import_all_functions_from_file(file_path=code_file_path)

        return policies_fns, generated_code

    def evaluate_policies(
        self,
        behavior: str,
        rollout_fn: callable,
        policies_fns: dict,
        ego_name: str,
        ego_visual_description: str,
        rollouts_output_dir: str,
        n_rollouts: int = 3,
        **kwargs,
    ):
        r_prefix = self._log_prefix()

        all_rollouts_states = []
        all_rollouts_images = []
        for rollout_idx in range(n_rollouts):
            logging.info(f"{r_prefix} Running rollout ({rollout_idx + 1}/{n_rollouts})...")

            rollout_subfolder = f"{rollouts_output_dir}/rollout_{rollout_idx}"

            # Perform rollout
            rollout_data, images, error = rollout_fn(**policies_fns, **kwargs)

            # Save rollout trajectory and images
            save_rollout(
                rollout_data=rollout_data,
                images=images,
                save_path=rollout_subfolder,
            )
            logging.debug(f"{r_prefix} Saved rollout to {rollout_subfolder}")

            if error is not None:
                raise RuntimeError(error)

            all_rollouts_states.append(uniform_sample(rollout_data, n_samples=20))
            all_rollouts_images.append(uniform_sample(images, n_samples=5))

        logging.info(f"{r_prefix} Evaluating rollout...")

        # Evaluate
        result = self.evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name=ego_name,
            ego_visual_description=ego_visual_description,
            rollouts_states=all_rollouts_states,
            rollouts_images=all_rollouts_images,
        )

        return result

    def _log_prefix(self) -> str:
        """Generate a structured log prefix like [B=0 | A=1]."""
        parts = []
        if self.behavior_idx is not None:
            parts.append(f"B={self.behavior_idx}")
        if self.align_idx is not None:
            parts.append(f"A={self.align_idx}")
        return f"[{' | '.join(parts)}]" if parts else ""
