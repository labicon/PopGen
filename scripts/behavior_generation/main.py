import os
import json
import importlib
import time
from functools import partial

import hydra
from hydra.utils import instantiate
from pqdm.processes import pqdm
from omegaconf import DictConfig, OmegaConf

from popgen.population import Behavior

OmegaConf.register_new_resolver(
    "basename",
    lambda s: s.split("/")[-1],
)
OmegaConf.register_new_resolver(
    "classname",
    lambda s: s.split(".")[-1],
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


BEHAVIOR_GENERATION_PROMPT = """You are designing player behaviors for a multi-agent environment.

Player Behavior Guidelines:
{item_description}

Environment Description:
{environment_description}

Generate a player behavior that respect these properties."""


def generate_behavior_group(
    group_idx: int,
    env_description: str,
    strategy_cfg: DictConfig,
    n_behaviors: int,
    output_dir: str,
) -> dict:
    """Generate a single group of behaviors."""
    group_output_dir = os.path.join(output_dir, f"group_{group_idx}")
    os.makedirs(group_output_dir, exist_ok=True)
    behavior_file_path = os.path.join(group_output_dir, "behaviors.json")

    # Check cache
    if os.path.exists(behavior_file_path):
        with open(behavior_file_path, "r") as f:
            behaviors_data = json.load(f)
        return {
            "group_idx": group_idx,
            "status": "cached",
            "n_behaviors": len(behaviors_data),
            "path": behavior_file_path,
        }

    # Initialize generator
    generator = instantiate(strategy_cfg, output_type=Behavior, output_dir=group_output_dir)

    input_prompt = BEHAVIOR_GENERATION_PROMPT.format(
        environment_description=env_description,
        item_description=Behavior.__doc__,
    )
    try:
        behaviors = generator.generate(
            input_prompt=input_prompt,
            num_items=n_behaviors,
        )

        behaviors_data = [b.model_dump() for b in behaviors]
        with open(behavior_file_path, "w") as f:
            json.dump(behaviors_data, f, indent=2)

        return {
            "group_idx": group_idx,
            "status": "success",
            "n_behaviors": len(behaviors),
            "path": behavior_file_path,
        }

    except Exception as e:
        return {
            "group_idx": group_idx,
            "status": "error",
            "error": str(e),
        }


def run_parallel_behavior_generation(cfg: DictConfig):
    """Generate multiple groups of behaviors in parallel."""
    # Get strategy name from _target_
    strategy_name = cfg.strategy._target_.split(".")[-1]

    # Create output directory
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logging.info("=" * 60)
    logging.info("PARALLEL BEHAVIOR GENERATION")
    logging.info("=" * 60)
    logging.info(f"Environment:      {cfg.env}")
    logging.info(f"Strategy:         {strategy_name}")
    logging.info(f"Num Groups:       {cfg.n_groups}")
    logging.info(f"Behaviors/Group:  {cfg.n_behaviors_per_group}")
    logging.info(f"Parallel Workers: {cfg.n_workers}")
    logging.info(f"Output:           {output_dir}")
    logging.info("-" * 60)

    # Load environment description
    env = importlib.import_module(f"popgen.envs.{cfg.env}")
    env_description = env.ENVIRONMENT_DESCRIPTION

    # Generate behavior groups in parallel
    logging.info(f"Generating {cfg.n_groups} behavior groups...")

    generate_fn = partial(
        generate_behavior_group,
        strategy_cfg=cfg.strategy,
        n_behaviors=cfg.n_behaviors_per_group,
        output_dir=output_dir,
    )

    args = [(i, env_description) for i in range(cfg.n_groups)]

    start_time = time.time()
    results = pqdm(
        args,
        generate_fn,
        n_jobs=cfg.n_workers,
        argument_type="args",
        desc="Generating behavior groups",
    )
    generation_time = time.time() - start_time

    for result in results:
        if result["status"] == "success":
            logging.info(f"[Group {result['group_idx']}] Generated {result['n_behaviors']} behaviors")
        elif result["status"] == "cached":
            logging.info(f"[Group {result['group_idx']}] Loaded {result['n_behaviors']} cached behaviors")
        else:
            logging.error(f"[Group {result['group_idx']}] Error: {result.get('error', 'Unknown')}")

    # Sort results by group index
    results.sort(key=lambda x: x["group_idx"])

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("GENERATION COMPLETE")
    logging.info("=" * 60)

    n_success = sum(1 for r in results if r["status"] in ("success", "cached"))
    n_cached = sum(1 for r in results if r["status"] == "cached")
    total_behaviors = sum(r.get("n_behaviors", 0) for r in results)

    logging.info(f"Groups:     {n_success}/{cfg.n_groups} successful ({n_cached} cached)")
    logging.info(f"Behaviors:  {total_behaviors} total")
    logging.info(f"Time:       {generation_time:.2f}s")

    # Save summary
    summary = {
        "env": cfg.env,
        "strategy": OmegaConf.to_container(cfg.strategy),
        "n_groups": cfg.n_groups,
        "n_behaviors_per_group": cfg.n_behaviors_per_group,
        "total_behaviors": total_behaviors,
        "total_generation_time": generation_time,
        "per_group_generation_time": generation_time / cfg.n_groups,
        "results": results,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved to {summary_path}")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_parallel_behavior_generation(cfg)


if __name__ == "__main__":
    main()
