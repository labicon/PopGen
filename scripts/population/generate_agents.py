import importlib
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pqdm.threads import pqdm

from popgen.population import (
    Behavior,
    BehaviorAdherenceEvaluator,
    LLMAgentGenerationPipeline,
    PolicyCodeAgentGenerationPipeline,
    PolicyCodeGenerator,
    RewardFnGenerator,
    RLAgentGenerationPipeline,
)
from popgen.utils.config import LitellmConfig

OmegaConf.register_new_resolver("basename", lambda s: s.split("/")[-1])
OmegaConf.register_new_resolver("classname", lambda s: s.split(".")[-1])
OmegaConf.register_new_resolver(
    "behavior_source",
    lambda bp, bt: os.path.splitext(os.path.basename(str(bp)))[0] if bp else bt.split(".")[-1],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

load_dotenv()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _build_job(
    cfg: DictConfig,
    behavior: Behavior,
    behavior_idx: int,
    trial_idx: int,
    flat_idx: int,
    result_path: str,
) -> dict[str, Any]:
    run_timestamp = _timestamp()
    output_dir = os.path.join(result_path, "agents", f"behavior_{behavior_idx}", run_timestamp)

    generation_cfg = OmegaConf.to_container(cfg.generation, resolve=True)
    evaluation_cfg = OmegaConf.to_container(cfg.evaluation, resolve=True)

    generation_type = str(generation_cfg["type"])
    max_attempts = int(
        generation_cfg.get("max_evaluation_attempts", evaluation_cfg["max_alignment_retries"])
        if generation_type == "llm_agent"
        else evaluation_cfg["max_alignment_retries"]
    )

    # Reserve 1 worker per run by default to avoid multiplicative fan-out.
    pipeline_workers = int(cfg.get("pipeline_workers", 1))

    return {
        "behavior": behavior.model_dump(),
        "behavior_idx": behavior_idx,
        "trial_idx": trial_idx,
        "flat_idx": flat_idx,
        "run_timestamp": run_timestamp,
        "output_dir": output_dir,
        "env": str(cfg.env),
        "seed": int(cfg.seed),
        "pipeline_workers": pipeline_workers,
        "generation_type": generation_type,
        "generation_cfg": generation_cfg,
        "evaluation_cfg": evaluation_cfg,
        "max_attempts": int(max_attempts),
    }


def _run_single_job(job: dict[str, Any]) -> dict[str, Any]:
    os.makedirs(job["output_dir"], exist_ok=True)

    try:
        behavior = Behavior.model_validate(job["behavior"])
        env = importlib.import_module(f"popgen.envs.{job['env']}")

        generation_cfg = job["generation_cfg"]
        evaluation_cfg = job["evaluation_cfg"]

        evaluator = BehaviorAdherenceEvaluator(
            model_config=LitellmConfig(model_name=evaluation_cfg["evaluator_model"]), num_workers=job["pipeline_workers"]
        )

        generation_type = job["generation_type"]
        if generation_type == "rl_reward":
            generator = RewardFnGenerator(
                model_config=LitellmConfig(model_name=generation_cfg["model_name"])
            )
            pipeline = RLAgentGenerationPipeline(
                generator=generator,
                evaluator=evaluator,
                env_module=env,
                output_dir=job["output_dir"],
            )
            generation_kwargs = {}
        elif generation_type == "policy_code":
            generator = PolicyCodeGenerator(
                model_config=LitellmConfig(model_name=generation_cfg["model_name"])
            )
            pipeline = PolicyCodeAgentGenerationPipeline(
                generator=generator,
                evaluator=evaluator,
                env_module=env,
                output_dir=job["output_dir"],
            )
            generation_kwargs = {}
        elif generation_type == "llm_agent":
            agent_cfg = LitellmConfig(model_name=generation_cfg["model_name"], generation_kwargs={"max_tokens": 2048})
            pipeline = LLMAgentGenerationPipeline(
                agent_model_config=agent_cfg,
                evaluator=evaluator,
                env_module=env,
                output_dir=job["output_dir"],
            )
            reserved_keys = {"type", "evaluate", "model_name", "max_evaluation_attempts", "n_generation_retries"}
            generation_kwargs = {k: v for k, v in generation_cfg.items() if k not in reserved_keys}
        else:
            raise ValueError(f"Unknown generation type: {generation_type}")

        success, summary = pipeline(
            behavior=behavior,
            num_attempts=job["max_attempts"],
            n_rollouts=int(evaluation_cfg["n_eval_rollouts"]),
            adherence_threshold=float(evaluation_cfg["adherence_threshold"]),
            n_generation_retries=int(generation_cfg.get("n_generation_retries", 3)),
            num_workers=int(job["pipeline_workers"]),
            generation_kwargs=generation_kwargs,
            evaluation_kwargs={"seed": int(job["seed"])},
        )

        result = {
            "behavior_idx": job["behavior_idx"],
            "trial_idx": job["trial_idx"],
            "flat_idx": job["flat_idx"],
            "run_timestamp": job["run_timestamp"],
            "seed": job["seed"],
            "pipeline_workers": job["pipeline_workers"],
            "output_dir": os.path.abspath(job["output_dir"]),
            "success": bool(success),
            "attempts_to_success": summary.get("attempts_to_success"),
            "error": None,
        }
    except Exception as e:
        result = {
            "behavior_idx": job["behavior_idx"],
            "trial_idx": job["trial_idx"],
            "flat_idx": job["flat_idx"],
            "run_timestamp": job["run_timestamp"],
            "seed": job["seed"],
            "pipeline_workers": job["pipeline_workers"],
            "output_dir": os.path.abspath(job["output_dir"]),
            "success": False,
            "attempts_to_success": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    return result


def run_pipeline(cfg: DictConfig):
    behavior_path = cfg.get("behavior_path", None)
    assert behavior_path, "behavior_path must be set"
    assert os.path.exists(behavior_path), f"Behavior file not found: {behavior_path}"

    result_path = cfg.output_dir
    os.makedirs(result_path, exist_ok=True)

    metadata_dir = os.path.join(result_path, "metadata", _timestamp())
    os.makedirs(metadata_dir, exist_ok=True)

    generation_type = cfg.generation.type

    logging.info("=" * 60)
    logging.info("FLATTENED MULTIAGENT RUNNER")
    logging.info("=" * 60)
    logging.info(f"Generation:  {generation_type}")
    logging.info(f"Behaviors:   {behavior_path}")
    logging.info(f"Environment: {cfg.env}")
    logging.info(f"Result path: {result_path}")
    logging.info("-" * 60)

    with open(behavior_path) as f:
        behaviors_data = json.load(f)
    behaviors = [Behavior.model_validate(b) for b in behaviors_data][: int(cfg.n_behaviors)]

    with open(os.path.join(metadata_dir, "behaviors.json"), "w") as f:
        json.dump([b.model_dump() for b in behaviors], f, indent=2)

    if not behaviors:
        raise ValueError("No behaviors loaded")

    n_trials = int(cfg.get("n_trials", 1))
    jobs: list[dict[str, Any]] = []
    flat_idx = 0
    for trial_idx in range(n_trials):
        for behavior_idx, behavior in enumerate(behaviors):
            jobs.append(
                _build_job(
                    cfg=cfg,
                    behavior=behavior,
                    behavior_idx=behavior_idx,
                    trial_idx=trial_idx,
                    flat_idx=flat_idx,
                    result_path=result_path,
                )
            )
            flat_idx += 1

    launch_workers = int(cfg.n_workers)
    logging.info(f"Launching {len(jobs)} runs in parallel (workers={launch_workers}, per-run pipeline_workers={cfg.get('pipeline_workers', 1)})")

    raw_results = pqdm(
        jobs,
        _run_single_job,
        n_jobs=launch_workers,
        desc="Running flattened jobs",
    )

    results: list[dict[str, Any]] = []
    for i, item in enumerate(raw_results):
        if isinstance(item, Exception):
            job = jobs[i]
            results.append(
                {
                    "behavior_idx": job["behavior_idx"],
                    "trial_idx": job["trial_idx"],
                    "flat_idx": job["flat_idx"],
                    "run_timestamp": job["run_timestamp"],
                    "seed": job["seed"],
                    "pipeline_workers": job["pipeline_workers"],
                    "output_dir": os.path.abspath(job["output_dir"]),
                    "success": False,
                    "attempts_to_success": None,
                    "error": str(item),
                }
            )
        else:
            results.append(item)

    n_success = sum(1 for r in results if r.get("success"))
    summary = {
        "generation_type": generation_type,
        "env": cfg.env,
        "result_path": os.path.abspath(result_path),
        "num_behaviors": len(behaviors),
        "n_trials": n_trials,
        "num_runs": len(jobs),
        "num_succeeded": n_success,
        "num_failed": len(jobs) - n_success,
        "runs": results,
    }

    with open(os.path.join(metadata_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("=" * 60)
    logging.info("RUN COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Success: {n_success}/{len(jobs)}")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
