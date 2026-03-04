"""Behavior similarity evaluator using semantic and lexical metrics."""

from __future__ import annotations

from typing import Any

import litellm
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

from popgen.utils.config import LitellmConfig

from .base import BaseEvaluator, EvalResult, calculate_stats


class SimilarityEvaluator(BaseEvaluator):
    """Evaluator for behavior similarity between two sets of texts.

    Computes:
    - Semantic similarity: Cosine similarity of sentence embeddings (index-wise).
    - Lexical similarity: ROUGE-L F1 scores (index-wise).
    """

    instance_plot_metrics = [
        ("semantic_similarity", "histogram"),
        ("lexical_similarity", "histogram"),
    ]
    aggregate_plot_metrics = ["semantic_similarity", "lexical_similarity"]
    key_plot_metrics = [("semantic_similarity", "Semantic Similarity")]

    def __init__(self, embedding_model: str | LitellmConfig, num_workers: int = 1, batch_size: int = 32):
        """Initialize the similarity evaluator.

        Args:
            embedding_model: LitellmConfig or model name string for the API embedding model
                             (e.g., "openrouter/google/gemini-embedding-001").
            num_workers: Number of workers.
            batch_size: Batch size for API embedding calls.
        """
        super().__init__(name="BehaviorSimilarity", num_workers=num_workers)
        if isinstance(embedding_model, str):
            embedding_model = LitellmConfig(model_name=embedding_model)
        self.embedding_config = embedding_model
        self.embedding_model = embedding_model.model_name  # kept for metadata
        self.batch_size = batch_size
        self._rouge_scorer = None

    @property
    def scorer(self) -> rouge_scorer.RougeScorer:
        """Lazy-load the ROUGE scorer."""
        if self._rouge_scorer is None:
            self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return self._rouge_scorer

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for a list of texts using the configured API embedder."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = litellm.embedding(
                model=self.embedding_config.model_name,
                input=batch,
                **self.embedding_config.get_completion_kwargs(),
            )
            all_embeddings.extend(
                item["embedding"] if isinstance(item, dict) else item.embedding
                for item in response.data
            )
        return np.array(all_embeddings)

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Compute similarity metrics for two lists of texts.

        Args:
            input_data: Dict with keys 'texts1' and 'texts2' containing lists of behaviors.
                        Behaviors are expected to be strings (descriptions) or dicts with 'description'.

        Returns:
            Dict with metrics.
        """
        texts1 = self._extract_texts(input_data.get("texts1", []))
        texts2 = self._extract_texts(input_data.get("texts2", []))

        if len(texts1) != len(texts2):
            raise ValueError(f"Input lists must have same length. Got {len(texts1)} and {len(texts2)}")

        n_texts = len(texts1)
        if n_texts == 0:
            return {"semantic_similarity_matrix": [], "lexical_similarity_matrix": [], "semantic_similarity": 0.0, "lexical_similarity": 0.0}

        # Compute semantic similarity
        semantic_score, semantic_output = self._compute_semantic_similarity(texts1, texts2)

        # Compute lexical similarity
        lexical_score, lexical_output = self._compute_lexical_similarity(texts1, texts2)

        return {
            "texts1": texts1,
            "texts2": texts2,
            "semantic_similarity": float(semantic_score),
            "semantic_embeddings": {
                "e1": semantic_output["embeddings1"],
                "e2": semantic_output["embeddings2"],
            },
            "semantic_similarity_matrix": semantic_output["similarity_matrix"],
            "lexical_similarity": float(lexical_score),
            "lexical_similarity_matrix": lexical_output["similarity_matrix"],
        }

    def _extract_texts(self, data: list[Any]) -> list[str]:
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "description" in item:
                texts.append(item["description"])
            else:
                texts.append(str(item))
        return texts

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate metrics."""
        for i, m in enumerate(instance_metrics):
            if self.save_path is not None and "semantic_embeddings" in m:
                for key, emb in m["semantic_embeddings"].items():
                    np.save(f"{self.save_path}/embeddings_{i}_{key}.npy", np.array(emb))

        return {
            "semantic_similarity": calculate_stats([m["semantic_similarity"] for m in instance_metrics]),
            "lexical_similarity": calculate_stats([m["lexical_similarity"] for m in instance_metrics]),
        }

    def _compute_semantic_similarity(self, texts1: list[str], texts2: list[str]) -> list[float]:
        # Batch encode together to minimise API round-trips
        combined_embeddings = self._get_embeddings(texts1 + texts2)
        split_idx = len(texts1)
        embeddings1 = combined_embeddings[:split_idx]
        embeddings2 = combined_embeddings[split_idx:]

        # Calculate cosine similarity matrix
        similarities = cosine_similarity(embeddings1, embeddings2)

        # Clip to [0, 1] range
        similarities = np.clip(similarities, 0, 1)

        # Compute pairwise similarities
        triu_indices = np.triu_indices(len(similarities), k=1)
        pairwise_sims = similarities[triu_indices]

        return (
            np.mean(pairwise_sims),
            {"embeddings1": embeddings1, "embeddings2": embeddings2, "similarity_matrix": similarities.tolist()},
        )

    def _compute_lexical_similarity(self, texts1: list[str], texts2: list[str]) -> list[list[float]]:
        matrix = []
        for t1 in texts1:
            row = []
            for t2 in texts2:
                score = self.scorer.score(t1, t2)
                row.append(score["rougeL"].fmeasure)
            matrix.append(row)

        # Compute pairwise similarities
        triu_indices = np.triu_indices(len(matrix), k=1)
        pairwise_sims = np.array(matrix)[triu_indices]

        return (np.mean(pairwise_sims), {"similarity_matrix": matrix})

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "Similarity Evaluator (Semantic + Lexical)",
                "semantic_scorer": self.embedding_model,
                "lexical_scorer": "ROUGE-L",
            }
        )

        return super().evaluate(inputs, metadata=metadata)
