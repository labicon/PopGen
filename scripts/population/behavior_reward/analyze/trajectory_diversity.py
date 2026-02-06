import json
import logging
import os
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from popgen.population.evaluation import BehaviorQualityEvaluator, DiversityEvaluator
from popgen.utils.config import LitellmConfig

from .base_experiment import TextAnalysisExperiment
from .visualization_utils import plot_behavior_tsne

logger = logging.getLogger(__name__)


class TrajectoryDiversityExperiment(TextAnalysisExperiment):
    @property
    def experiment_name(self) -> str:
        return "trajectory_diversity"

    def evaluate_population(self, desired_behaviors: list[dict], collected_data: list[Any], env_name: str, strategy: str):
        # Filter None
        valid_behaviors = [b for b in collected_data if b is not None and b.get("description")]
        if not valid_behaviors:
            logger.warning("No valid behaviors to analyze.")
            return

        logger.info("Computing diversity metrics for the population...")

        # Extract descriptions
        analyzed_descriptions = [b.get("description", "") for b in valid_behaviors]

        # Diversity Evaluation
        diversity_evaluator = DiversityEvaluator(embedding_model=self.cfg.get("embedding_model", "Qwen/Qwen3-Embedding-8B"), save_path=self.output_path, num_workers=1)

        diversity_results = diversity_evaluator.evaluate([{"texts": analyzed_descriptions}], metadata={"env_name": env_name, "strategy": strategy})

        # Save raw stats
        diversity_evaluator.save_results(diversity_results, os.path.join(self.output_path, "diversity_scores.json"))

        # Extract Metrics
        semantic_diversity = float(diversity_results.overall_metrics["semantic_diversity"]["mean"])
        lexical_diversity = float(diversity_results.overall_metrics["lexical_diversity"]["mean"])

        instance_metrics = diversity_results.instance_metrics[0]
        semantic_sim = np.array(instance_metrics.get("semantic_similarity", np.eye(len(analyzed_descriptions))))
        lexical_sim = np.array(instance_metrics.get("lexical_similarity", np.eye(len(analyzed_descriptions))))

        embeddings = diversity_evaluator.model.encode(analyzed_descriptions, convert_to_numpy=True)

        # Plot
        plot_behavior_tsne(
            behaviors=valid_behaviors,
            output_path=self.output_path,
            embeddings=embeddings,
            semantic_similarity_matrix=semantic_sim,
            lexical_similarity_matrix=lexical_sim,
            semantic_diversity=semantic_diversity,
            lexical_diversity=lexical_diversity,
            perplexity=self.cfg.get("perplexity", 5.0),
            env_name=env_name,
            strategy=strategy,
        )

        # Quality Analysis
        logger.info("Computing quality metrics...")
        judge_config = LitellmConfig(model_name=self.cfg.get("judge_model", "openai/gpt-5-mini"))
        quality_evaluator = BehaviorQualityEvaluator(model_config=judge_config)
        quality_inputs = [{"behavior": f"{b['name']}: {b['description']}", "environment_description": self.env_description} for b in valid_behaviors]

        quality_results = quality_evaluator.evaluate(quality_inputs, metadata={"env_name": env_name})
        quality_evaluator.save_results(quality_results, os.path.join(self.output_path, "quality_scores.json"))

        qm = quality_results.overall_metrics
        quality_summary = {
            "quality_relevance": qm["relevance"]["mean"],
            "quality_realism": qm["realism"]["mean"],
            "quality_specificity": qm["specificity"]["mean"],
            "quality_mean": (qm["relevance"]["mean"] + qm["realism"]["mean"] + qm["specificity"]["mean"]) / 3,
        }

        # Log
        logger.info("=" * 40)
        logger.info(f"Sem/Lex Diversity: {semantic_diversity:.4f} / {lexical_diversity:.4f}")
        logger.info(f"Quality Mean:      {quality_summary['quality_mean']:.4f}")
        logger.info("=" * 40)

        # Final Report
        report = {"metrics": {"semantic_diversity": semantic_diversity, "lexical_diversity": lexical_diversity, **quality_summary}}
        with open(os.path.join(self.experiment_root, "diversity_report.json"), "w") as f:
            json.dump(report, f, indent=2)


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup Loggers
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    exp = TrajectoryDiversityExperiment(cfg)
    exp.run()


if __name__ == "__main__":
    main()
