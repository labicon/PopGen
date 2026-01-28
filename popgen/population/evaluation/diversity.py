"""Behavior diversity evaluator using semantic and lexical metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import EvalResult, calculate_stats
from .similarity import SimilarityEvaluator


class DiversityEvaluator(SimilarityEvaluator):
    """Evaluator for behavior diversity using semantic and lexical metrics.

    Computes:
    - Semantic diversity: 1 - mean(pairwise cosine similarities) of sentence embeddings
    - Lexical diversity: 1 - mean(pairwise ROUGE-L F1 scores)
    """

    instance_plot_metrics = [
        ("semantic_diversity", "histogram"),
        ("lexical_diversity", "histogram"),
    ]
    aggregate_plot_metrics = ["semantic_diversity", "lexical_diversity", "unique_words", "vocabulary_richness"]
    key_plot_metrics = [("semantic_diversity", "Semantic Diversity")]

    def __init__(self, embedding_model: str, num_workers: int = 1, save_path: str | None = None, batch_size: int = 32):
        """Initialize the diversity evaluator.

        Args:
            embedding_model: Name of the sentence transformer model for embeddings.
            num_workers: Number of workers (kept at 1 since we batch embeddings).
            save_path: Optional path to save the embeddings as a .npy file.
            batch_size: Batch size for sentence transformer encoding.
        """
        super().__init__(embedding_model=embedding_model, num_workers=num_workers, batch_size=batch_size)
        self.name = "BehaviorDiversity"
        self.save_path = save_path

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Compute diversity metrics for a batch of texts from one prompt.

        Args:
            input_data: Dict with key 'texts' containing a list of behavior dicts,
                        each with keys 'name' and 'description'

        Returns:
            Dict with semantic_diversity, lexical_diversity, and supporting metrics
        """
        texts = self._extract_texts(input_data.get("texts", []))
        n_texts = len(texts)

        # Compute word statistics across all texts
        all_words = []
        for behavior in texts:
            all_words.extend(behavior.lower().split())
        response_length = len(all_words)
        unique_words = set(all_words)
        num_unique_words = len(unique_words)
        vocabulary_richness = num_unique_words / response_length if response_length > 0 else 0.0

        if n_texts < 2:
            return {
                "texts": texts,
                "semantic_diversity": 0.0,
                "semantic_similarity_matrix": np.array([]),
                "semantic_embeddings": [],
                "lexical_diversity": 0.0,
                "lexical_similarity_matrix": np.array([]),
                "unique_words": num_unique_words,
                "vocabulary_richness": vocabulary_richness,
            }

        # Compute semantic diversity
        semantic_similarity, semantic_output = self._compute_semantic_similarity(texts, texts)

        # Compute lexical diversity
        lexical_similarity, lexical_output = self._compute_lexical_similarity(texts, texts)

        return {
            "texts": texts,
            "semantic_diversity": float(1.0 - semantic_similarity),
            "semantic_similarity_matrix": semantic_output["similarity_matrix"],
            "semantic_embeddings": semantic_output["embeddings1"],
            "lexical_diversity": float(1.0 - lexical_similarity),
            "lexical_similarity_matrix": lexical_output["similarity_matrix"],
            "unique_words": num_unique_words,
            "vocabulary_richness": vocabulary_richness,
        }

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate diversity metrics across all batches.

        Args:
            instance_metrics: List of dicts from compute_instance_metric

        Returns:
            Dict with stats for each metric across all batches
        """
        for i, m in enumerate(instance_metrics):
            if self.save_path is not None and "semantic_embeddings" in m:
                np.save(f"{self.save_path}/embeddings_{i}.npy", np.array(m["semantic_embeddings"]))

        return {
            "semantic_diversity": calculate_stats([m["semantic_diversity"] for m in instance_metrics]),
            "lexical_diversity": calculate_stats([m["lexical_diversity"] for m in instance_metrics]),
            "unique_words": calculate_stats([m["unique_words"] for m in instance_metrics]),
            "vocabulary_richness": calculate_stats([m["vocabulary_richness"] for m in instance_metrics]),
        }

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        """Evaluate a list of texts and rollouts."""
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "Diversity Evaluator (Semantic + Lexical)",
                "semantic_scorer": self.embedding_model,
                "lexical_scorer": "ROUGE-L",
            }
        )

        result = super().evaluate(inputs, metadata=metadata)

        # remove embeddings from instance metrics to save space
        for m in result.instance_metrics:
            if "semantic_embeddings" in m:
                del m["semantic_embeddings"]

        return result
