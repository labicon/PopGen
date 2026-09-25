import json
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv

from popgen.population import Behavior

load_dotenv(override=True)


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
    datefmt="%H:%M:%S",
)


def _resolve_behavior_file_path(cfg: DictConfig) -> str:
    explicit_path = cfg.get("behavior_path", None)
    if explicit_path:
        return str(explicit_path)

    import time
    time_ = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return os.path.join("data", "behaviors", f"{cfg.env}__{cfg.behavior._target_.split('.')[-1]}__{time_}.json")


def run_behavior_generation(cfg: DictConfig) -> None:
    behavior_file_path = _resolve_behavior_file_path(cfg)
    os.makedirs(os.path.dirname(behavior_file_path), exist_ok=True)

    logging.info("=" * 60)
    logging.info("BEHAVIOR GENERATION ONLY")
    logging.info("=" * 60)
    logging.info(f"Behavior strategy: {cfg.behavior._target_}")
    logging.info(f"Environment: {cfg.env}")
    logging.info(f"Output: {behavior_file_path}")
    logging.info("-" * 60)

    if os.path.exists(behavior_file_path):
        logging.info(f"Loading cached behaviors from {behavior_file_path}")
        with open(behavior_file_path, "r") as f:
            behaviors_data = json.load(f)
        behaviors = [Behavior.model_validate(b) for b in behaviors_data]
    else:
        env = __import__(f"popgen.envs.{cfg.env}", fromlist=["*"])
        behavior_generator = instantiate(cfg.behavior, output_type=Behavior)
        behaviors = behavior_generator.generate(
            input_prompt=env.ENVIRONMENT_DESCRIPTION,
            num_items=cfg.n_behaviors,
        )
        behaviors_data = [b.model_dump() for b in behaviors]

    with open(behavior_file_path, "w") as f:
        json.dump(behaviors_data, f, indent=2)

    logging.info(f"Saved {len(behaviors)} behavior(s) to {behavior_file_path}")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    run_behavior_generation(cfg)


if __name__ == "__main__":
    main()
