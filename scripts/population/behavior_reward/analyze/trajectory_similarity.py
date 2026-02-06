import json
import logging
import os
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from popgen.population.evaluation import SimilarityEvaluator

from .base_experiment import TextAnalysisExperiment
from .visualization_utils import plot_similarity_comparison

logger = logging.getLogger(__name__)


class TrajectorySimilarityExperiment(TextAnalysisExperiment):
    @property
    def experiment_name(self) -> str:
        return "trajectory_similarity"

    def evaluate_population(self, desired_behaviors: list[dict], collected_data: list[Any], env_name: str, strategy: str):
        # Prepare inputs
        desired_texts = []
        analyzed_texts = []
        behavior_names = []

        # Ensure we handle missing
        for i, (desired, analyzed) in enumerate(zip(desired_behaviors, collected_data)):
            # If default scanning was used, desired might be None or derived
            d_desc = desired.get("description", "") if desired else ""
            a_desc = analyzed.get("description", "") if analyzed else ""

            desired_texts.append(d_desc)
            analyzed_texts.append(a_desc)
            behavior_names.append(desired.get("name", f"Behavior {i}") if desired else f"Behavior {i}")

        logger.info("Computing similarity metrics...")

        similarity_evaluator = SimilarityEvaluator(embedding_model=self.cfg.get("embedding_model", "Qwen/Qwen3-Embedding-8B"), num_workers=1)

        similarity_results = similarity_evaluator.evaluate([{"texts1": desired_texts, "texts2": analyzed_texts}], metadata={"env_name": env_name, "strategy": strategy})

        similarity_evaluator.save_results(similarity_results, os.path.join(self.output_path, "similarity_scores.json"))

        # Log summary
        qm = similarity_results.overall_metrics
        logger.info("=" * 40)
        logger.info(f"Semantic Similarity: {qm['semantic_similarity']['mean']:.4f}")
        logger.info(f"Lexical Similarity:  {qm['lexical_similarity']['mean']:.4f}")
        logger.info("=" * 40)

        # Plot
        instance = similarity_results.instance_metrics[0]
        semantic_scores = instance["semantic_similarity_scores"]
        lexical_scores = instance["lexical_similarity_scores"]

        embeddings_desired = similarity_evaluator.model.encode(desired_texts, convert_to_numpy=True)
        embeddings_analyzed = similarity_evaluator.model.encode(analyzed_texts, convert_to_numpy=True)

        plot_similarity_comparison(
            behavior_names=behavior_names,
            output_path=self.output_path,
            embeddings_desired=embeddings_desired,
            embeddings_analyzed=embeddings_analyzed,
            semantic_scores=semantic_scores,
            lexical_scores=lexical_scores,
            semantic_mean=float(qm["semantic_similarity"]["mean"]),
            lexical_mean=float(qm["lexical_similarity"]["mean"]),
            env_name=env_name,
            strategy=strategy,
        )

        # Final Report
        report = {
            "metrics": {"semantic_similarity": qm["semantic_similarity"]["mean"], "lexical_similarity": qm["lexical_similarity"]["mean"]},
            "input_behaviors": desired_texts,
            "analyzed_behaviors": analyzed_texts,
        }
        with open(os.path.join(self.experiment_root, "similarity_report.json"), "w") as f:
            json.dump(report, f, indent=2)


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup Loggers
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    exp = TrajectorySimilarityExperiment(cfg)
    exp.run()


if __name__ == "__main__":
    main()
