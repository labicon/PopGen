
import os
import json
import logging
import time

from .generation.generator import Generator
from .evaluator import BehaviorEvaluator, EvaluationResult

from popgen.utils.imports import import_all_functions_from_file
from popgen.utils.logging import save_rollout

logging.basicConfig(level=logging.INFO)

class FeedbackGenerationPipeline:
    """Pipeline for generating diverse behaviors, policies, and rewards."""
    
    def __init__(
            self, 
            generator: Generator,
            evaluator: BehaviorEvaluator,
            rollout_fn: callable,
            output_dir: str,
        ):
        self.generator: Generator = generator
        self.evaluator: BehaviorEvaluator = evaluator
        self.output_dir = output_dir
        self.item_type = self.generator.generation_type
        self.rollout_fn = rollout_fn

        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(
            self, 
            behavior: str,
            behavior_index: int,
            num_attempts: int = 5,
            **kwargs
    ) -> dict:
        feedbacks = []

        # Initialize indices for logging
        self.behavior_idx = behavior_index
        self.align_idx = None
        self.rollout_idx = None

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
                **kwargs
            )
            generation_latency = time.time() - generation_start

            rollout_output_dir = f"{attempt_output_dir}/rollout"
            os.makedirs(rollout_output_dir, exist_ok=True)

            evaluation_start = time.time()
            try:
                evaluation_result = self.evaluate_policies(
                    behavior=behavior,
                    rollout_fn=self.rollout_fn,
                    policies_fns=policies_fns,
                    rollout_output_dir=rollout_output_dir,
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
                "generation": {"attempts": self.align_idx + 1, "item": generated_item, "latency": generation_latency},
                "evaluation": {"rollout_path": rollout_output_dir, "result": evaluation_result.model_dump(), "latency": evaluation_latency},
            }

            evaluation_output_path = f"{attempt_output_dir}/evaluation.json"
            with open(evaluation_output_path, "w") as f:
                json.dump(output, f, indent=2)

            if evaluation_result.evaluation:
                logging.info(f"{self.log_prefix} SUCCESS – Alignment succeeded on attempt {attempt + 1}")
                return True, output
            else:
                logging.warning(f"{self.log_prefix} FAILURE – Alignment failed on attempt {attempt + 1}. Reason: {evaluation_result.reasoning[100:]}.")
                feedbacks.append((generated_item, evaluation_result.reasoning))
                if attempt < num_attempts - 1:
                    logging.info(f"{self.log_prefix} Retrying alignment (Attempt {attempt + 2}/{num_attempts})...")

        logging.error(f"{self.log_prefix} All {num_attempts} attempts exhausted. Alignment failed.")
        return False, output
    
    def generate_policies(
            self, 
            behavior: str, 
            feedbacks: list,
            generation_output_dir: str,
            **kwargs
        ):
        generated_code = self.generator.generate(
            behavior=behavior,
            feedbacks=feedbacks,
            **kwargs
        )

        logging.debug(f"{self.log_prefix} Generated {self.item_type}:\n{generated_code.code}")

        # save generated code to file
        os.makedirs(generation_output_dir, exist_ok=True)
        code_file_path = f"{generation_output_dir}/{self.item_type}.py"
        with open(code_file_path, "w") as f:
            f.write(generated_code.code)
        logging.debug(f"{self.log_prefix} Saved generated {self.item_type} to {code_file_path}")

        # Import policy functions from generated code
        policies_fns = import_all_functions_from_file(file_path=code_file_path)

        return policies_fns, generated_code.code

    def evaluate_policies(
            self,
            behavior: str,
            rollout_fn: callable,
            policies_fns: dict,
            rollout_output_dir: str,
            **rollout_kwargs
        ):
        r_prefix = self._log_prefix()

        logging.info(f"{r_prefix} Running rollout...")

        # Perform rollout
        rollout_data, images, error = rollout_fn(**policies_fns, **rollout_kwargs)

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
        
        logging.info(f"{r_prefix} Evaluating rollout...")

        # trajectory is str representation of rollout_data
        trajectory = "\n".join(str(state) for state in rollout_data)
        images_path = f"{rollout_output_dir}/rollout.gif"

        # Evaluate
        result = self.evaluator.evaluate_trajectory_and_images(
            trajectory,
            [images_path],
            behavior
        )

        return result

    def _log_prefix(self) -> str:
        """Generate a structured log prefix like [B=0 | A=1 | R=2]."""
        parts = []
        if self.behavior_idx is not None:
            parts.append(f"B={self.behavior_idx}")
        if self.align_idx is not None:
            parts.append(f"A={self.align_idx}")
        if self.rollout_idx is not None:
            parts.append(f"R={self.rollout_idx}")
        return f"[{' | '.join(parts)}]" if parts else ""
