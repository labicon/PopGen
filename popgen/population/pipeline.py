import json
import logging
import os
import shutil
import traceback
from abc import ABC, abstractmethod
from time import time
from typing import Any

from pqdm.processes import pqdm

from popgen.utils.config import LitellmConfig
from popgen.utils.decorators import track_time
from popgen.utils.imports import import_function_from_file, load_entrypoint
from popgen.utils.logging import save_rollout, uniform_sample, write_code_file, write_json, write_model_output_record
from popgen.utils.warnings import configure_warning_filters

from .evaluation import BehaviorAdherenceEvaluator
from .generation import BaseGenerator
from .types import Behavior

logging.basicConfig(level=logging.INFO)
configure_warning_filters()

def combine_feedback(instance_metrics: list[dict[str, Any]]) -> str:
    """Combine feedback from failing rollouts into a string for the next generation attempt."""

    def _build_feedback_str(i: int, reasoning: str, feedback: str, adherence: float) -> str:
        success = adherence == 1
        s = f"- [Rollout #{i}] {'Success' if success else 'Failure'}:\n"
        s += f"  Reason: {reasoning}\n"
        if not success:
            s += f"  Feedback: {feedback}\n"
        return s

    bullets = [
        _build_feedback_str(i, m["reasoning"], m["feedback"], m["adherence"]) 
        for i, m in enumerate(instance_metrics) 
    ]
    if not bullets:
        return "No specific feedback provided."
    return "Feedback from rollouts:\n" + "\n".join(bullets)

def _average_latency(values: list[Any]) -> Any:
    """Average homogeneous lists of numbers or dicts of numbers."""
    if not values:
        return 0.0

    first = values[0]
    if isinstance(first, dict):
        keys = first.keys()
        out = {}
        for key in keys:
            key_values = [v[key] for v in values if isinstance(v, dict) and key in v]
            out[key] = sum(key_values) / len(key_values) if key_values else 0.0
        return out

    return sum(float(v) for v in values) / len(values)


def _latency_total(latency: Any) -> float:
    """Reduce nested latency structures (dict/list/float) to one scalar."""
    if isinstance(latency, dict):
        return sum(_latency_total(v) for v in latency.values())
    if isinstance(latency, list):
        return sum(_latency_total(v) for v in latency)
    if latency is None:
        return 0.0
    return float(latency)

def _format_exc_last_frame(exc: BaseException) -> str:
    """Return only the deepest traceback frame + exception message."""
    tb = exc.__traceback__
    if tb is None:
        return f"{type(exc).__name__}: {exc}"
    last = traceback.extract_tb(tb)[-1]
    return (
        f'File "{last.filename}", line {last.lineno}, in {last.name}\n'
        f"    {last.line}\n"
        f"{type(exc).__name__}: {exc}"
    )


def _build_rollout_error_feedback(rollouts: list[dict[str, Any] | Exception]) -> str | None:
    """Return structured feedback when any rollout fails."""
    errors = []
    for rollout_idx, result in enumerate(rollouts):
        if isinstance(result, Exception):
            errors.append(f"Rollout #{rollout_idx}:\n{_format_exc_last_frame(result)}")
        elif result.get("error") is not None:
            errors.append(f"Rollout #{rollout_idx}:\n{result['error']}")

    if not errors:
        return None

    feedback = "Rollout execution failed:\n"
    feedback += "\n".join(f"- {err}" for err in errors)
    feedback += "\nPlease fix the code implementation so rollouts execute without errors."
    return feedback


def _load_policy_and_env(policy_type: str, create_env_fn: callable, scripted_policy_fn: callable = None, **kwargs) -> dict[str, Any]:
    """Load policy artifacts and environment for one behavior attempt."""
    if policy_type == "PolicyCode":
        from popgen.agents import PolicyCodeAgent
        from popgen.utils.imports import import_all_functions_from_file

        policies = import_all_functions_from_file(kwargs["file_path"])
        policies = {name: PolicyCodeAgent(func) for name, func in policies.items()}
        if "policy" not in policies:
            raise KeyError("Expected generated policy file to define a callable named 'policy'.")
        ego_policy = policies["policy"]
        env = create_env_fn(num_envs=1)
    elif policy_type == "RewardFn":
        from pathlib import Path

        from benchmarl.experiment import Experiment

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
        ego_group = "ego" if "ego" in experiment.group_policies else "agents"
        ego_policy = experiment.group_policies[ego_group]
        env = experiment.test_env
    elif policy_type == "LLMAgent":
        from popgen.agents import LLMAgent

        config_path = kwargs["config_path"]
        with open(config_path) as f:
            config = json.load(f)

        model_config = LitellmConfig(**config["model_config"])
        action_type = load_entrypoint(config["action_type_module"])

        ego_policy = LLMAgent(
            behavior=config["behavior"],
            model_config=model_config,
            environment_description=config["environment_description"],
            action_type=action_type,
            horizon_length=config.get("horizon_length", 1),
            include_trajectory_history=config.get("include_trajectory_history", False),
            feedbacks=config.get("feedbacks", []),
        )
        env = create_env_fn(num_envs=1)
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    policies = [ego_policy]
    if scripted_policy_fn is not None:
        teammate_policy = scripted_policy_fn(env)
        policies.append(teammate_policy)

    return {"env": env, "policies": policies}


@track_time("rollout")
def _rollout_worker(args: dict[str, Any]) -> dict[str, Any]:
    """Run one rollout and save artifacts."""
    # Child workers may start a fresh interpreter depending on start method.
    configure_warning_filters(force=True)
    rollout_fn = args["rollout_fn"]
    artifacts_dict = _load_policy_and_env(
        policy_type=args["policy_type"],
        create_env_fn=args["create_env_fn"],
        scripted_policy_fn=args["scripted_policy_fn"],
        **args.get("loading_kwargs", {}),
    )

    try:
        rollout_data, images, error = rollout_fn(artifacts_dict["policies"], env=artifacts_dict["env"], **args["rollout_kwargs"])
    except Exception as e:
        return {"error": _format_exc_last_frame(e), "states": None, "images": None}

    save_rollout(
        rollout_data=rollout_data,
        images=images,
        save_path=args["rollout_subfolder"],
    )

    if error is not None:
        error_str = _format_exc_last_frame(error) if isinstance(error, Exception) else str(error)
        return {"error": error_str, "states": None, "images": None}

    return {
        "error": None,
        "states": uniform_sample(rollout_data, n_samples=20),
        "images": uniform_sample(images, n_samples=5),
    }


class AgentGenerationPipeline(ABC):
    """Base pipeline that handles a single behavior per run."""

    policy_type: str

    def __init__(
        self,
        generator: BaseGenerator | None,
        evaluator: BehaviorAdherenceEvaluator,
        env_module: Any,
        output_dir: str,
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.env_module = env_module
        self.output_dir = output_dir
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

    def _generate_with_retries(
        self,
        behavior: Behavior,
        attempt_dir: str,
        n_generation_retries: int,
        n_rollouts: int,
        feedbacks: list[tuple[str, str]] | None = None,
        num_workers: int = 8,
        generation_kwargs: dict | None = None,
        evaluation_kwargs: dict | None = None,
    ) -> tuple[Any, list[Any], bool, int, dict[str, Any]]:
        """Run generation + rollout + evaluation for one behavior, with retries."""
        feedbacks = list(feedbacks) if feedbacks is not None else []
        generation_kwargs = generation_kwargs or {}
        evaluation_kwargs = evaluation_kwargs or {}

        generated_item = None
        rollouts = []
        rollout_latencies = []
        generation_latency = 0.0
        success = False


        for gen_retry in range(n_generation_retries + 1):
            # Step 1: generation
            try:
                generated_item, loading_kwargs, generation_latency = self._generate_policy(
                    behavior=behavior,
                    feedbacks=feedbacks,
                    output_dir=attempt_dir,
                    **generation_kwargs,
                )
            except Exception as e:
                generation_error = f"Generation or training failed: {e}. Please try again."
                logging.warning(generation_error)
                feedbacks.append(("", generation_error))
                continue

            # Step 2: rollouts
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
            rollout_error_feedback = _build_rollout_error_feedback(rollouts)
            success = rollout_error_feedback is None
            if success:
                break
            else:
                logging.warning(f"[Gen Retry {gen_retry}] {rollout_error_feedback}")

                prev_code = getattr(generated_item, "code", "")
                feedbacks.append((prev_code, rollout_error_feedback))

        latencies = {
            "generation": generation_latency,
            "rollout": rollout_latencies,
        }
            
        return generated_item, rollouts, success, gen_retry, latencies

    def __call__(
        self,
        behavior: Behavior,
        num_attempts: int = 5,
        n_rollouts: int = 3,
        adherence_threshold: float = 1.0,
        n_generation_retries: int = 0,
        num_workers: int = 8,
        generation_kwargs: dict | None = None,
        evaluation_kwargs: dict | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Run generation + rollout + evaluation for one behavior.

        `n_generation_retries` controls extra generation attempts inside each
        alignment attempt when generation/training fails or rollouts error out.
        """
        start_time = time()
        generation_kwargs = generation_kwargs or {}
        evaluation_kwargs = evaluation_kwargs or {}

        feedbacks: list[tuple[str, str]] = []
        best_adherence = 0.0
        best_feedback: tuple[str, str] | None = None
        last_adherence = 0.0
        success = False
        attempts_to_success: int | None = None
        generated_item: Any = None
        attempt_summaries: list[dict[str, Any]] = []

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

            # Step 1: generation & rollouts (with retries):
            generated_item, rollouts, gen_success, gen_retries, gen_latencies = self._generate_with_retries(
                behavior=behavior,
                attempt_dir=attempt_dir,
                n_generation_retries=n_generation_retries,
                n_rollouts=n_rollouts,
                feedbacks=context_feedbacks,
                num_workers=num_workers,
                generation_kwargs=generation_kwargs,
                evaluation_kwargs=evaluation_kwargs,
            )

            # Step 2: evaluation
            eval_elapsed = 0.0
            try:
                eval_result, eval_elapsed = self._evaluate_policy(
                    behavior=behavior,
                    rollouts=rollouts,
                    behavior_dir=attempt_dir,
                )
            except Exception as e:
                eval_result = None
                evaluation_feedback = f"Evaluation failed: {e}"
                logging.warning(f"[Attempt {attempt + 1}] {evaluation_feedback}")
                feedbacks.append(("", evaluation_feedback))

            adherence = eval_result.overall_metrics["adherence"] if eval_result is not None else 0.0
            rollout_elapsed = _latency_total(gen_latencies["rollout"])
            success = adherence >= adherence_threshold
            if success:
                attempts_to_success = attempt + 1
                logging.info(f"[Attempt {attempt + 1}] SUCCESS (adherence={adherence:.2f})")
            elif eval_result is not None:
                overall_feedback = combine_feedback(eval_result.instance_metrics)
                logging.warning(f"[Attempt {attempt + 1}] FAILURE (adherence={adherence:.2f})")
                logging.debug(f"Overall feedback:\n{overall_feedback}")
                generated_artifact = getattr(generated_item, "code", "")
                current_feedback = (generated_artifact, overall_feedback)
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
                "latencies": {
                    "generation": gen_latencies["generation"],
                    "rollout": rollout_elapsed,
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
            "num_workers": num_workers,
            "success": success,
            "attempts_to_success": attempts_to_success,
            "n_attempts": n_attempts,
            "avg_latencies": {
                "generation": _average_latency([a["latencies"]["generation"] for a in attempt_summaries]),
                "rollout": _average_latency([a["latencies"]["rollout"] for a in attempt_summaries]),
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
        if not isinstance(gen_code, Exception):

            def _train_once() -> None:
                compute_reward = import_function_from_file(code_file_path, "compute_reward")
                self.env_module.train_fn(compute_reward, output_dir)

            try:
                _, training_latency = track_time("training")(_train_once)()
            except Exception as e:
                gen_code = e
                training_latency = 0.0

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
    ):
        super().__init__(
            generator=None,
            evaluator=evaluator,
            env_module=env_module,
            output_dir=output_dir,
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
