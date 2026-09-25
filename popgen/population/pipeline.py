import json
import logging
import os
import shutil
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any

import litellm
from pqdm.processes import pqdm

from popgen.agents import EgoTeammateRouter, MultiPolicyRouter
from popgen.agents.reward_policy import (
    latest_reward_training_log,
    load_reward_rollout_artifacts,
)
from popgen.population.evaluation import BehaviorAdherenceEvaluator
from popgen.population.feedback import gather_feedback
from popgen.population.generation import BaseGenerator
from popgen.population.types import Behavior
from popgen.utils.config import LitellmConfig
from popgen.utils.decorators import track_time
from popgen.utils.exceptions import TrainingFailure, format_exc_last_frame
from popgen.utils.imports import import_function_from_file, load_entrypoint
from popgen.utils.logging import save_rollout, uniform_sample, write_code_file, write_json, write_model_output_record
from popgen.utils.rewards import DEFAULT_REWARD_CLAMP, sanitize_reward_fn
from popgen.utils.run_names import (
    build_population_run_name,
    extract_attempt_idx_from_path,
    extract_behavior_idx_from_path,
    extract_run_date_from_path,
    infer_env_name,
)
from popgen.utils.stats import average_values
from popgen.utils.warnings import configure_warning_filters

logging.basicConfig(level=logging.INFO)
configure_warning_filters()

DEFAULT_EVAL_ROLLOUTS = 5
# Extra RL trainings allowed per alignment attempt after a slow training
# failure. 0 => a slow failure ends the attempt; the next attempt starts from a
# fresh directory and a genuinely new reward.
DEFAULT_TRAINING_RETRIES = 0
# A training failure only counts against DEFAULT_TRAINING_RETRIES if training
# ran at least this long. Faster crashes (bad reward shape, for instance) are
# cheap and stay on the generation-retry budget.
DEFAULT_MIN_EXPENSIVE_TRAINING_SECONDS = 120.0
EVAL_ROLLOUT_STATES = 30
EVAL_ROLLOUT_IMAGES = 3


def load_policy_and_env(policy_type: str, create_env_fn: callable, scripted_policy_fn: callable = None, reload_context=None, device: str = "cpu", **kwargs) -> tuple[Any, Any]:
    """Load policy artifacts and environment for one behavior attempt.

    Always returns an ``(env, policy)`` pair. When ``scripted_policy_fn`` is
    given, ``policy`` is a router wrapping ego + teammate; otherwise it is the
    ego policy alone.
    """
    if policy_type == "PolicyCode":
        from popgen.agents import PolicyCodeAgent
        from popgen.utils.imports import import_all_functions_from_file

        policies = import_all_functions_from_file(kwargs["file_path"])
        policies = {name: PolicyCodeAgent(func) for name, func in policies.items()}
        if "policy" not in policies:
            raise KeyError("Expected generated policy file to define a callable named 'policy'.")
        ego_policy = policies["policy"]
    elif policy_type == "RewardFn":
        return load_reward_rollout_artifacts(
            save_dir=kwargs["save_dir"],
            scripted_policy_fn=scripted_policy_fn,
            reload_context=reload_context,
            device=device,
        )
    elif policy_type == "LLMAgent":
        from popgen.agents import LLMAgent

        config_path = kwargs["config_path"]
        with open(config_path) as f:
            config = json.load(f)

        ego_policy = LLMAgent(
            behavior=config["behavior"],
            model_config=LitellmConfig(**config["model_config"]),
            environment_description=config["environment_description"],
            action_type=load_entrypoint(config["action_type_module"]),
            horizon_length=config.get("horizon_length", 1),
            include_trajectory_history=config.get("include_trajectory_history", False),
            feedbacks=config.get("feedbacks", []),
        )
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    # Create the rollout environment
    env = create_env_fn(num_envs=1)

    # If a scripted policy is provided, load it and wrap with the ego policy for rollouts.
    if scripted_policy_fn is not None:
        teammate_policy = scripted_policy_fn(env)
        teammate_policies = list(teammate_policy) if isinstance(teammate_policy, (list, tuple)) else [teammate_policy]
        policy = MultiPolicyRouter(policies={"ego": [ego_policy], "teammate": teammate_policies})
    else:
        policy = ego_policy

    return env, policy


@track_time("rollout")
def _rollout_worker(args: dict[str, Any]) -> dict[str, Any]:
    """Run one rollout and save artifacts."""
    # Child workers may start a fresh interpreter depending on start method.
    configure_warning_filters(force=True)

    rollout_fn = args["rollout_fn"]
    env, policy = load_policy_and_env(
        policy_type=args["policy_type"],
        create_env_fn=args["create_env_fn"],
        scripted_policy_fn=args["scripted_policy_fn"],
        reload_context=args.get("reload_context"),
        **args.get("loading_kwargs", {}),
    )

    try:
        rollout_data, images, error = rollout_fn(policy, env=env, **args["rollout_kwargs"])
    except Exception as e:
        try:
            env.close()
        except Exception:
            pass
        return {"error": format_exc_last_frame(e), "states": None, "images": None}

    save_rollout(
        rollout_data=rollout_data,
        images=images,
        save_path=args["rollout_subfolder"],
    )

    if error is not None:
        error_str = format_exc_last_frame(error) if isinstance(error, Exception) else str(error)
        try:
            env.close()
        except Exception:
            pass
        return {"error": error_str, "states": None, "images": None}

    result = {
        "error": None,
        "states": uniform_sample(rollout_data, n_samples=EVAL_ROLLOUT_STATES),
        "images": uniform_sample(images, n_samples=EVAL_ROLLOUT_IMAGES),
    }
    try:
        env.close()
    except Exception:
        pass
    return result


class AgentGenerationPipeline(ABC):
    """Base pipeline that handles a single behavior per run."""

    policy_type: str

    def __init__(
        self,
        generator: BaseGenerator | None,
        evaluator: BehaviorAdherenceEvaluator,
        env_module: Any,
        output_dir: str,
        advisor: LitellmConfig | None = None,
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.env_module = env_module
        self.output_dir = output_dir
        self.advisor = advisor
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def _generate_policy(
        self,
        behavior: Behavior,
        feedbacks: list[tuple[str, str]],
        output_dir: str,
        **kwargs,
    ) -> tuple[Any, dict[str, Any], Any]:
        """Generate one policy artifact and loading kwargs."""

    def generate_and_rollout(
        self,
        behavior: Behavior,
        attempt_dir: str,
        n_generation_retries: int,
        n_rollouts: int,
        feedbacks: list[tuple[str, str]] | None = None,
        num_workers: int = 8,
        generation_kwargs: dict | None = None,
        evaluation_kwargs: dict | None = None,
        n_training_retries: int = DEFAULT_TRAINING_RETRIES,
        min_expensive_training_seconds: float = DEFAULT_MIN_EXPENSIVE_TRAINING_SECONDS,
    ) -> tuple[Any, list[Any], bool, int, dict[str, Any], dict[str, Any]]:
        """Run generation + rollout + evaluation for one behavior, with retries.

        Cheap failures (generation, import, rollout) may retry up to
        ``n_generation_retries`` times. Training failures that ran for at least
        ``min_expensive_training_seconds`` are charged to the much smaller
        ``n_training_retries`` budget instead, since each one costs tens of
        minutes of GPU time. Only ``RLAgentGenerationPipeline`` can produce
        them, so this is inert for policy-code and LLM-agent generation.
        """
        feedbacks = list(feedbacks) if feedbacks is not None else []
        n_context_feedbacks = len(feedbacks)
        generation_kwargs = generation_kwargs or {}
        evaluation_kwargs = evaluation_kwargs or {}

        generated_item = None
        rollouts = []
        rollout_latencies = []
        generation_latency = 0.0
        success = False
        training_failures = 0
        gen_retry = 0

        for gen_retry in range(n_generation_retries + 1):
            failure: BaseException | None = None
            try:
                generated_item, loading_kwargs, generation_latency = self._generate_policy(
                    behavior=behavior,
                    feedbacks=feedbacks,
                    output_dir=attempt_dir,
                    **generation_kwargs,
                )
            except Exception as e:
                failure = e
            else:
                if isinstance(generated_item, Exception):
                    failure = generated_item

            if failure is not None:
                # TrainingFailure pre-renders its own message (it is returned,
                # not raised, so it has no traceback of its own).
                detail = str(failure) if isinstance(failure, TrainingFailure) else format_exc_last_frame(failure)
                generation_error = f"Generation or training failed:\n{detail}\nPlease try again."
                logging.warning(f"[Gen Retry {gen_retry}] {generation_error}")
                feedbacks.append(("", generation_error))

                if isinstance(failure, TrainingFailure) and failure.elapsed >= min_expensive_training_seconds:
                    training_failures += 1
                    if training_failures > n_training_retries:
                        logging.warning(f"Training retry budget exhausted after {training_failures} slow training failure(s) (last ran {failure.elapsed:.0f}s); abandoning this attempt.")
                        break
                continue

            try:
                rollouts, rollout_latencies = self._run_rollouts(
                    behavior_dir=attempt_dir,
                    loading_kwargs=loading_kwargs,
                    n_rollouts=n_rollouts,
                    num_workers=num_workers,
                    **evaluation_kwargs,
                )
            except Exception as e:
                rollouts, rollout_latencies = [e], []

            # If error in rollback -> retry
            errors = []
            for rollout_idx, result in enumerate(rollouts):
                if isinstance(result, Exception):
                    errors.append(f"Rollout #{rollout_idx}:\n{format_exc_last_frame(result)}")
                elif result.get("error") is not None:
                    errors.append(f"Rollout #{rollout_idx}:\n{result['error']}")

            if not errors:
                success = True
                break

            rollout_error_feedback = "Rollout execution failed:\n"
            rollout_error_feedback += "\n".join(f"- {err}" for err in errors)
            rollout_error_feedback += "\nPlease fix the code implementation so rollouts execute without errors."
            logging.warning(f"[Gen Retry {gen_retry}] {rollout_error_feedback}")

            prev_code = getattr(generated_item, "code", "")
            feedbacks.append((prev_code, rollout_error_feedback))

        latencies = {
            "generation": generation_latency,
            "rollout": rollout_latencies,
        }
        # Feedback accumulated across retries lives on this (copied) list and
        # would otherwise be discarded with the frame; hand it back so the next
        # attempt's context carries the real error rather than a placeholder.
        retry_info = {
            "feedbacks": feedbacks[n_context_feedbacks:],
            "training_failures": training_failures,
        }

        return generated_item, rollouts, success, gen_retry, latencies, retry_info

    def __call__(
        self,
        behavior: Behavior,
        num_attempts: int = 5,
        n_rollouts: int = DEFAULT_EVAL_ROLLOUTS,
        adherence_threshold: float = 1.0,
        n_generation_retries: int = 0,
        num_workers: int = 8,
        generation_kwargs: dict | None = None,
        evaluation_kwargs: dict | None = None,
        n_training_retries: int = DEFAULT_TRAINING_RETRIES,
        min_expensive_training_seconds: float = DEFAULT_MIN_EXPENSIVE_TRAINING_SECONDS,
    ) -> tuple[bool, dict[str, Any]]:
        """Run generation + rollout + evaluation for one behavior.

        `n_generation_retries` controls extra generation attempts inside each
        alignment attempt when generation/training fails or rollouts error out.
        `n_training_retries` is the separate, much smaller budget for failures
        of the RL training step itself (see `generate_and_rollout`).
        """
        start_time = time()
        generation_kwargs = generation_kwargs or {}
        evaluation_kwargs = evaluation_kwargs or {}

        generated_item: Any = None
        feedbacks: list[tuple[str, str]] = []
        best_adherence = 0.0
        best_feedback: tuple[str, str] | None = None
        last_adherence = 0.0
        attempts_to_success: int | None = None
        attempt_summaries: list[dict[str, Any]] = []
        success = False

        for attempt in range(num_attempts):
            if success:
                break

            # Create attempt directory for artifacts and logs
            attempt_dir = os.path.join(self.output_dir, f"attempt_{attempt}")
            os.makedirs(attempt_dir, exist_ok=True)
            logging.info(f"[Attempt {attempt + 1}/{num_attempts}] Processing behavior: {behavior.name}")

            # Build context: best attempt + last attempt (deduplicated)
            context_feedbacks = []
            if best_feedback is not None:
                context_feedbacks.append((best_feedback[0], f"[Best attempt so far (adherence={best_adherence:.2f})]\n{best_feedback[1]}"))
            if feedbacks and feedbacks[-1] is not best_feedback:
                context_feedbacks.append((feedbacks[-1][0], f"[Last attempt (adherence={last_adherence:.2f})]\n{feedbacks[-1][1]}"))

            write_json(
                os.path.join(attempt_dir, "feedback.json"),
                [{"code": code, "feedback": fb} for code, fb in context_feedbacks],
            )

            # Step 1: generation & rollouts (with retries):
            generated_item, rollouts, gen_success, gen_retries, gen_latencies, retry_info = self.generate_and_rollout(
                behavior=behavior,
                attempt_dir=attempt_dir,
                n_generation_retries=n_generation_retries,
                n_rollouts=n_rollouts,
                feedbacks=context_feedbacks,
                num_workers=num_workers,
                generation_kwargs=generation_kwargs,
                evaluation_kwargs=evaluation_kwargs,
                n_training_retries=n_training_retries,
                min_expensive_training_seconds=min_expensive_training_seconds,
            )

            # Step 2: evaluation
            if not rollouts:
                # Nothing was produced to judge: every retry failed before
                # rollouts ran. Evaluating an empty set used to raise, and the
                # resulting "division by zero" was all the next attempt saw.
                # Note this is deliberately `not rollouts`, not `not
                # gen_success`: rollouts that ran and errored are still worth
                # judging -- the evaluator turns them into usable feedback.
                eval_result = None
                eval_elapsed = 0.0
                retry_feedbacks = retry_info["feedbacks"]
                last_feedback = retry_feedbacks[-1] if retry_feedbacks else ("", "Generation failed with no recorded error.")
                logging.warning(f"[Attempt {attempt + 1}] No rollouts were produced; skipping evaluation.")
                feedbacks.append(last_feedback)
                last_adherence = 0.0
            else:
                try:
                    eval_result, eval_elapsed = self._evaluate_policy(
                        behavior=behavior,
                        rollouts=rollouts,
                        behavior_dir=attempt_dir,
                    )
                except Exception as e:
                    eval_result = None
                    eval_elapsed = 0.0
                    evaluation_feedback = f"Evaluation failed: {format_exc_last_frame(e)}"
                    logging.warning(f"[Attempt {attempt + 1}] {evaluation_feedback}")
                    feedbacks.append(("", evaluation_feedback))
                    last_adherence = 0.0

            adherence = eval_result.overall_metrics["adherence"] if eval_result is not None else 0.0

            success = adherence >= adherence_threshold
            evaluation_error = eval_result is None
            if success:
                logging.info(f"[Attempt {attempt + 1}] SUCCESS (adherence={adherence:.2f})")
                attempts_to_success = attempt + 1
            elif (not success) and (not evaluation_error):
                logging.warning(f"[Attempt {attempt + 1}] FAILURE (adherence={adherence:.2f})")
                generated_artifact = getattr(generated_item, "code", None)
                overall_feedback = gather_feedback(
                    eval_result.instance_metrics,
                    code_attempt=generated_artifact,
                    behavior=str(behavior),
                    environment_description=self.env_module.ENVIRONMENT_DESCRIPTION,
                    generation_type=self.policy_type,
                    advisor_config=self.advisor,
                )
                logging.debug(f"Overall feedback:\n{overall_feedback}")
                current_feedback = (generated_artifact or "", overall_feedback)
                if adherence > best_adherence:
                    best_adherence = adherence
                    best_feedback = current_feedback
                last_adherence = adherence
                feedbacks.append(current_feedback)

            latency_record = {
                "generation": gen_latencies["generation"],
                "rollout": gen_latencies["rollout"],
                "evaluation": eval_elapsed,
            }
            write_json(os.path.join(attempt_dir, "latency.json"), latency_record)

            attempt_summary = {
                "attempt": attempt + 1,
                "generation_success": gen_success,
                "success": success,
                "adherence": adherence,
                "generation_retries": gen_retries,
                "training_retries": retry_info["training_failures"],
                "latencies": {
                    "generation": gen_latencies["generation"],
                    "rollout": sum(float(v) for v in gen_latencies["rollout"]),
                    "evaluation": eval_elapsed,
                },
            }
            attempt_summaries.append(attempt_summary)

        total_time = time() - start_time
        n_attempts = len(attempt_summaries)
        overall_summary = {
            "behavior": behavior.model_dump(),
            "adherence_threshold": adherence_threshold,
            "n_generation_retries": n_generation_retries,
            "n_training_retries": n_training_retries,
            "num_workers": num_workers,
            "success": success,
            "attempts_to_success": attempts_to_success,
            "n_attempts": n_attempts,
            "avg_latencies": {
                "generation": average_values([a["latencies"]["generation"] for a in attempt_summaries]),
                "rollout": average_values([a["latencies"]["rollout"] for a in attempt_summaries]),
                "evaluation": (sum(a["latencies"]["evaluation"] for a in attempt_summaries) / n_attempts if n_attempts else 0.0),
            },
            "total_time": total_time,
            "attempt_summaries": attempt_summaries,
        }

        write_json(os.path.join(self.output_dir, "summary.json"), overall_summary)

        return success, overall_summary

    def _run_rollouts(
        self,
        behavior_dir: str,
        loading_kwargs: dict[str, Any],
        n_rollouts: int,
        num_workers: int,
        seed: int = 0,
        **rollout_kwargs,
    ) -> tuple[list[dict[str, Any] | Exception], list[float]]:
        """Run rollouts for one behavior."""
        output_path = os.path.join(behavior_dir, "rollouts")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        rollout_args = []
        for rollout_idx in range(n_rollouts):
            rollout_seed = seed + rollout_idx * 1000
            rollout_args.append(
                {
                    "rollout_fn": self.env_module.rollout_fn,
                    "policy_type": self.policy_type,
                    "create_env_fn": self.env_module.create_env,
                    "scripted_policy_fn": self.env_module.ScriptedPolicy,
                    # module-level function -> picklable by reference for spawned workers
                    "reload_context": getattr(self.env_module, "get_cooperator_reload_context", None),
                    "loading_kwargs": loading_kwargs,
                    "rollout_subfolder": os.path.join(output_path, f"rollout_{rollout_idx}"),
                    "rollout_kwargs": {**rollout_kwargs, "seed": rollout_seed},
                }
            )

        logging.info(f"Running {n_rollouts} rollouts with {num_workers} workers...")
        rollout_results = pqdm(
            rollout_args,
            _rollout_worker,
            n_jobs=num_workers,
            desc="Running rollouts",
        )

        rollouts: list[dict[str, Any] | Exception] = []
        latencies: list[float] = []
        for result in rollout_results:
            if isinstance(result, Exception):
                rollouts.append(result)
                latencies.append(0.0)
            else:
                rollout, elapsed = result
                rollouts.append(rollout)
                latencies.append(elapsed)
        return rollouts, latencies

    @track_time("evaluation")
    def _evaluate_policy(
        self,
        behavior: Behavior,
        rollouts: list[dict[str, Any] | Exception],
        behavior_dir: str,
    ) -> Any:
        """Evaluate one behavior using its rollout outputs."""

        environment_description = self.env_module.ENVIRONMENT_DESCRIPTION
        eval_inputs = []
        for result in rollouts:
            rollout_states = []
            rollout_images = []
            if isinstance(result, Exception):
                rollout_states = f"[ERROR] Rollout: {result}"
                rollout_images = None
            elif result["error"] is not None:
                rollout_states = f"[ERROR] Rollout: {result['error']}"
                rollout_images = None
            else:
                rollout_states = result["states"]
                rollout_images = result["images"]

            eval_inputs.append(
                {
                    "behavior": str(behavior),
                    "environment_description": environment_description,
                    "rollout_states": rollout_states,
                    "rollout_images": rollout_images,
                    "serialize_state_fn": getattr(self.env_module, "serialize_state_fn", None),
                }
            )

        eval_result = self.evaluator.evaluate(eval_inputs)

        for rollout_idx, instance_metric in enumerate(eval_result.instance_metrics):
            instance_metric["rollout_data_path"] = os.path.abspath(os.path.join(behavior_dir, "rollouts", f"rollout_{rollout_idx}", "rollout_data.jsonl"))

        write_json(
            os.path.join(behavior_dir, "evaluation.json"),
            {
                "instance_metrics": eval_result.instance_metrics,
                "overall_metrics": eval_result.overall_metrics,
                "metadata": eval_result.metadata,
            },
        )

        return eval_result


class PolicyCodeAgentGenerationPipeline(AgentGenerationPipeline):
    """Single-behavior pipeline for generated policy code."""

    policy_type = "PolicyCode"

    def _generate_policy(
        self,
        behavior: Behavior,
        feedbacks: list[tuple[str, str]],
        output_dir: str,
        **kwargs,
    ) -> tuple[Any, dict[str, Any], float]:
        try:
            gen_code, gen_latency = track_time("generate")(self.generator.generate)(
                behavior=str(behavior),
                feedbacks=feedbacks,
                environment_description=self.env_module.ENVIRONMENT_DESCRIPTION,
                examples=None,
                **kwargs,
            )
        except Exception as e:
            gen_code, gen_latency = e, 0.0

        code_file_path = os.path.join(output_dir, "policy.py")
        write_code_file(code_file_path, gen_code)
        write_model_output_record(output_dir, gen_code)

        return gen_code, {"file_path": code_file_path}, gen_latency


class RLAgentGenerationPipeline(AgentGenerationPipeline):
    """Single-behavior pipeline for reward generation + RL training."""

    policy_type = "RewardFn"

    def __init__(self, *args, train_kwargs: dict | None = None, reward_clamp: float | None = DEFAULT_REWARD_CLAMP, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_kwargs = dict(train_kwargs or {})
        self.reward_clamp = reward_clamp

    def _generate_policy(
        self,
        behavior: Behavior,
        feedbacks: list[tuple[str, str]],
        output_dir: str,
        **kwargs,
    ) -> tuple[Any, dict[str, Any], dict[str, float]]:
        environment_description = self.env_module.ENVIRONMENT_DESCRIPTION
        reward_api_reference = getattr(self.env_module, "REWARD_API_REFERENCE", None)
        examples = [self.env_module.REWARD_EXAMPLE]

        try:
            gen_code, reward_generation_latency = track_time("generate")(self.generator.generate)(
                behavior=str(behavior),
                feedbacks=feedbacks,
                environment_description=environment_description,
                reward_api_reference=reward_api_reference,
                examples=examples,
                **kwargs,
            )
        except Exception as e:
            gen_code, reward_generation_latency = e, 0.0

        code_file_path = os.path.join(output_dir, "reward.py")
        write_code_file(code_file_path, gen_code)
        write_model_output_record(output_dir, gen_code)

        training_latency = 0.0
        compute_reward = None
        run_name = Path(output_dir).name
        if not isinstance(gen_code, Exception):
            try:
                env_name = infer_env_name(self.env_module)
                run_date = extract_run_date_from_path(output_dir)
                behavior_idx = extract_behavior_idx_from_path(output_dir)
                attempt_idx = extract_attempt_idx_from_path(output_dir)
                run_name = build_population_run_name(
                    env_name=env_name,
                    run_date=run_date,
                    behavior_idx=behavior_idx,
                    attempt_idx=attempt_idx,
                )
            except (TypeError, ValueError):
                run_name = Path(output_dir).name

            # Importing the generated module is cheap and its failures (syntax
            # error, missing `compute_reward`) are trivially fixable, so keep it
            # out of the training try-block: those must stay on the cheap
            # generation-retry budget, not the expensive training one.
            try:
                compute_reward = import_function_from_file(code_file_path, "compute_reward")
                compute_reward = sanitize_reward_fn(compute_reward, self.reward_clamp)
            except Exception as e:
                gen_code = e

        if not isinstance(gen_code, Exception) and compute_reward is not None:
            from popgen.utils.filesystem import find_reward_checkpoints

            # Retries share this attempt directory, so a previous retry's
            # checkpoints would otherwise satisfy the "did training produce
            # anything?" check below.
            checkpoints_before = {p.resolve() for p in find_reward_checkpoints(output_dir)}
            started = time()
            try:
                self.env_module.train_fn(
                    reward_fn=compute_reward,
                    output_dir=output_dir,
                    run_name=run_name,
                    **self.train_kwargs,
                )
                training_latency = time() - started
            except Exception as e:
                # track_time() would discard the elapsed time on failure, and we
                # need it to tell a seconds-long crash from an hours-long one.
                gen_code = TrainingFailure(e, elapsed=time() - started, training_log=latest_reward_training_log(output_dir))
                training_latency = 0.0

        if not isinstance(gen_code, Exception) and compute_reward is not None:
            new_checkpoints = [p for p in find_reward_checkpoints(output_dir) if p.resolve() not in checkpoints_before]
            if not new_checkpoints:
                latest_log = latest_reward_training_log(output_dir)
                hint = f" Latest training log: {latest_log}. " if latest_log is not None else " "
                gen_code = TrainingFailure(
                    FileNotFoundError(f"No checkpoints produced in {output_dir}.{hint}Training likely failed before checkpointing."),
                    elapsed=training_latency,
                    training_log=latest_log,
                )

        return gen_code, {"save_dir": output_dir}, {"reward_generation": reward_generation_latency, "training": training_latency}


class LLMAgentGenerationPipeline(AgentGenerationPipeline):
    """Single-behavior pipeline for LLM-agent config generation."""

    policy_type = "LLMAgent"

    def __init__(
        self,
        agent_model_config: LitellmConfig,
        evaluator: BehaviorAdherenceEvaluator,
        env_module: Any,
        output_dir: str,
        advisor: LitellmConfig | None = None,
    ):
        super().__init__(
            generator=None,
            evaluator=evaluator,
            env_module=env_module,
            output_dir=output_dir,
            advisor=None,  # LLM Agents use rollout reasoning directly, no advisor
        )
        self.agent_model_config = agent_model_config

    def _generate_policy(
        self,
        behavior: Behavior,
        feedbacks: list[tuple[str, str]],
        output_dir: str,
        horizon_length: int = 1,
        include_trajectory_history: bool = False,
    ) -> tuple[Any, dict[str, Any], float]:
        environment_description = self.env_module.ENVIRONMENT_DESCRIPTION
        action_type = self.env_module.ActionType
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
            "feedbacks": feedbacks,
        }

        config_file_path = os.path.join(output_dir, "llm_agent_config.json")
        write_json(config_file_path, config)
        write_model_output_record(output_dir, generated_item=None, extra_record={"config_path": config_file_path, **config})

        return behavior.description, {"config_path": config_file_path}, 0.0


__all__ = [
    "AgentGenerationPipeline",
    "PolicyCodeAgentGenerationPipeline",
    "RLAgentGenerationPipeline",
    "LLMAgentGenerationPipeline",
]
