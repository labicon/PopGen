import abc
import importlib
import json
import logging
import os
from typing import Any, Optional, Union

from omegaconf import DictConfig

from popgen.population.generation import RolloutSynthetizer

from .analysis_utils import find_successful_attempt, run_rollout

logger = logging.getLogger(__name__)


class BaseAnalysisExperiment(abc.ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiment_name_str = self.experiment_name
        self.output_path = cfg.get("output_path", os.path.join(cfg.output_dir, f"{self.experiment_name_str}_plots"))
        os.makedirs(self.output_path, exist_ok=True)

    @property
    @abc.abstractmethod
    def experiment_name(self) -> str:
        """Name of the experiment (used for output folders)"""
        pass

    def setup(self):
        """Optional setup (load models, envs, etc)"""
        pass

    @abc.abstractmethod
    def process_behavior(self, behavior_ind: int, behavior_data: dict, successful_attempt: str) -> Any:
        """Process a single behavior/attempt and return data for evaluation."""
        pass

    @abc.abstractmethod
    def evaluate_population(self, desired_behaviors: list[dict], collected_data: list[Any], env_name: str, strategy: str):
        """Run population-level evaluation."""
        pass

    def run(self):
        logger.info("=" * 60)
        logger.info(f"{self.experiment_name.upper()} ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Environment: {self.cfg.env}")
        logger.info(f"Experiment Dir:   {self.cfg.output_dir}")
        if hasattr(self.cfg, "llm_model"):
            logger.info(f"LLM Model:   {self.cfg.llm_model}")

        if not os.path.exists(self.cfg.output_dir):
            logger.error(f"Experiment directory {self.cfg.output_dir} does not exist.")
            return

        # Resolve actual experiment root (handling timestamped folder structure)
        self.experiment_root = self.cfg.output_dir

        # If behavior_0 is missing in root, look for it in subdirectories
        if not os.path.exists(os.path.join(self.experiment_root, "behavior_0")):
            try:
                subdirs = [d for d in os.listdir(self.experiment_root) if os.path.isdir(os.path.join(self.experiment_root, d))]
                # Filter distinct folders (ignore plots, hydra, hidden)
                valid_subdirs = [d for d in subdirs if not d.endswith("_plots") and not d.startswith(".") and d != ".hydra"]

                for d in valid_subdirs:
                    if os.path.exists(os.path.join(self.experiment_root, d, "behavior_0")):
                        self.experiment_root = os.path.join(self.experiment_root, d)
                        logger.info(f"Auto-detected experiment root subdirectory: {self.experiment_root}")
                        break
            except Exception as e:
                logger.warning(f"Error while scanning subdirectories: {e}")

        # Load Behaviors
        behaviors_json_path = os.path.join(self.experiment_root, "behaviors.json")

        desired_behavior_list = []
        is_labeled_population = False

        if os.path.exists(behaviors_json_path):
            with open(behaviors_json_path) as f:
                desired_behavior_list = json.load(f)
            logger.info(f"Loaded {len(desired_behavior_list)} desired behaviors from JSON.")
            is_labeled_population = True
        else:
            logger.info("No behaviors.json found. Scanned directories will be used.")

        self.setup()

        collected_data = []  # The results of process_behavior
        final_desired_behaviors = []

        if is_labeled_population:
            # Case 1: We have a target list of behaviors (e.g. Similarity Analysis)
            collected_data = [None] * len(desired_behavior_list)
            final_desired_behaviors = desired_behavior_list

            for i, b_data in enumerate(desired_behavior_list):
                b_dir = f"behavior_{i}"
                b_path = os.path.join(self.experiment_root, b_dir)

                attempt_path, _ = find_successful_attempt(b_path, min_adherence=0.6)

                if not attempt_path:
                    logger.warning(f"No successful attempt found for {b_dir}")
                    continue  # Leave as None

                logger.info(f"Processing {b_dir}...")
                data = self.process_behavior(i, b_data, attempt_path)
                collected_data[i] = data
        else:
            # Case 2: We just have folders (e.g. Diversity Analysis might handle this)
            behavior_dirs = [d for d in os.listdir(self.experiment_root) if d.startswith("behavior_")]
            try:
                behavior_dirs.sort(key=lambda x: int(x.split("_")[1]) if len(x.split("_")) > 1 else 0)
            except (ValueError, IndexError):
                behavior_dirs.sort()

            for b_dir in behavior_dirs:
                b_path = os.path.join(self.experiment_root, b_dir)
                if not os.path.isdir(b_path):
                    continue

                attempt_path, eval_data = find_successful_attempt(b_path)

                if attempt_path:
                    # Construct dummy behavior data if not available
                    b_data = {"name": b_dir, "description": ""}
                    # Try to get from eval_data
                    if eval_data and "behavior" in eval_data:
                        raw_b = eval_data["behavior"]
                        if isinstance(raw_b, dict):
                            b_data = raw_b
                        else:
                            b_data = {"name": b_dir, "description": str(raw_b)}

                    logger.info(f"Processing {b_dir}...")
                    try:
                        ind = int(b_dir.split("_")[1])
                    except (ValueError, IndexError):
                        ind = len(collected_data)

                    data = self.process_behavior(ind, b_data, attempt_path)

                    # Store
                    collected_data.append(data)
                    final_desired_behaviors.append(b_data)

        # Prepare info for evaluation
        run_dir_name = os.path.basename(os.path.normpath(self.experiment_root))
        parts = run_dir_name.split("__")
        # heuristic env/strategy extraction
        env_from_dir = parts[0] if len(parts) >= 1 else "UnknownEnv"
        strategy = parts[1] if len(parts) >= 2 else "UnknownStrategy"

        env_name = self.cfg.env or env_from_dir

        self.evaluate_population(final_desired_behaviors, collected_data, env_name, strategy)


class TextAnalysisExperiment(BaseAnalysisExperiment):
    """Base for experiments that require rolling out and analyzing behavior text."""

    def setup(self):
        self.env_module = importlib.import_module(f"popgen.envs.{self.cfg.env}")
        self.analyzer = RolloutSynthetizer(model=self.cfg.llm_model)

        # Load env description
        if hasattr(self.env_module, "ENVIRONMENT_DESCRIPTION"):
            self.env_description = self.env_module.ENVIRONMENT_DESCRIPTION
        else:
            self.env_description = f"Environment: {self.cfg.env}"

    @property
    def rollout_subdir_prefix(self) -> str:
        return "analyze_rollout"

    def process_behavior(self, behavior_ind: int, behavior_data: dict, successful_attempt: str) -> dict[str, Any]:
        """Returns dict with 'name' and 'description' of analyzed behavior."""
        # 1. Check cache
        analysis_json_path = os.path.join(successful_attempt, "diversity_analysis.json")
        if os.path.exists(analysis_json_path):
            try:
                with open(analysis_json_path) as f:
                    existing = json.load(f)
                if "analyzed_behavior" in existing:
                    logger.info("Loaded analysis from file.")
                    return existing["analyzed_behavior"]
            except Exception as e:
                logger.warning(f"Failed to load existing analysis: {e}")

        # 2. Rollout
        rollout_fn = self.env_module.rollout_fn

        logger.info("Running rollout and analysis...")
        try:
            states, images = run_rollout(rollout_fn, successful_attempt, self.cfg.get("n_rollouts", 5), self.rollout_subdir_prefix)

            if not states:
                logger.warning("No states collected")
                return {}

            # 3. Analyze
            ego_name = "agent0"
            ego_visual = self.env_module.EGO_VISUAL_DESCRIPTION

            analysis_result = self.analyzer.analyze_rollouts(ego_name=ego_name, ego_visual_description=ego_visual, env_description=self.env_description, rollouts_states=states, rollouts_images=images)

            result = analysis_result.model_dump()
            logger.info(f"Analyzed: {result.get('name', 'Unknown')}")

            # Save
            entry = {"behavior_id": f"behavior_{behavior_ind}", "attempt_dir": successful_attempt, "original_behavior": behavior_data, "analyzed_behavior": result}
            with open(analysis_json_path, "w") as f:
                json.dump(entry, f, indent=2)

            return result

        except Exception as e:
            logger.error(f"Error processing {successful_attempt}: {e}")
            import traceback

            traceback.print_exc()
            return {}
