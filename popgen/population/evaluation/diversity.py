"""Behavior diversity evaluator using semantic and lexical metrics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseEvaluator, EvalResult, calculate_stats


class DiversityEvaluator(BaseEvaluator):
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

    def __init__(self, embedding_model: str, num_workers: int = 1, save_path: str | None = None):
        """Initialize the diversity evaluator.

        Args:
            embedding_model: Name of the sentence transformer model for embeddings.
            num_workers: Number of workers (kept at 1 since we batch embeddings).
            save_path: Optional path to save the embeddings as a .npy file.
        """
        super().__init__(name="BehaviorDiversity", num_workers=num_workers)
        self.embedding_model = embedding_model
        self.save_path = save_path
        self._model = None
        self._rouge_scorer = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    @property
    def scorer(self) -> rouge_scorer.RougeScorer:
        """Lazy-load the ROUGE scorer."""
        if self._rouge_scorer is None:
            self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return self._rouge_scorer

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Compute diversity metrics for a batch of texts from one prompt.

        Args:
            input_data: Dict with key 'texts' containing a list of behavior dicts,
                        each with keys 'name' and 'description'

        Returns:
            Dict with semantic_diversity, lexical_diversity, and supporting metrics
        """
        texts = input_data["texts"]
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
                "semantic_similarity": None,
                "lexical_diversity": 0.0,
                "lexical_similarity": None,
                "unique_words": 0,
                "vocabulary_richness": 0.0,
            }

        # Compute semantic diversity
        semantic_diversity, pairwise_sim, embeddings = self._compute_semantic_diversity(texts)

        # Compute lexical diversity
        lexical_diversity, pairwise_rouge = self._compute_lexical_diversity(texts)

        return {
            "texts": texts,
            "semantic_diversity": semantic_diversity,
            "semantic_similarity": pairwise_sim,
            "semantic_embeddings": embeddings,
            "lexical_diversity": lexical_diversity,
            "lexical_similarity": pairwise_rouge,
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

    def _compute_semantic_diversity(self, texts: list[str]) -> tuple[float, float]:
        """Compute semantic diversity using sentence embeddings.

        Args:
            texts: List of behavior text representations

        Returns:
            Tuple of (diversity_score, mean_pairwise_similarity)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        similarity_matrix = cosine_similarity(embeddings)

        # Get upper triangle (excluding diagonal) for pairwise similarities
        triu_indices = np.triu_indices(len(texts), k=1)
        pairwise_sims = similarity_matrix[triu_indices]

        # Clip negative similarities to 0 to avoid inflating diversity
        pairwise_sims = np.clip(pairwise_sims, 0, 1)

        mean_sim = float(pairwise_sims.mean())
        diversity_score = 1.0 - mean_sim

        return diversity_score, similarity_matrix.tolist(), embeddings

    def _compute_lexical_diversity(self, texts: list[str]) -> tuple[float, float]:
        """Compute lexical diversity using ROUGE-L.

        Args:
            texts: List of behavior text representations

        Returns:
            Tuple of (diversity_score, mean_pairwise_rouge)
        """
        n = len(texts)
        pairwise_scores = []

        for i in range(n):
            for j in range(i + 1, n):
                score = self.scorer.score(texts[i], texts[j])
                pairwise_scores.append(score["rougeL"].fmeasure)

        mean_rouge = float(np.mean(pairwise_scores))
        diversity_score = 1.0 - mean_rouge

        # construct the full pairwise ROUGE matrix
        full_matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    full_matrix[i, j] = 1.0
                elif i < j:
                    full_matrix[i, j] = pairwise_scores[idx]
                    full_matrix[j, i] = pairwise_scores[idx]
                    idx += 1

        return diversity_score, full_matrix.tolist()

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
