"""ROUGE-L-based diversity evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEvaluator, EvalResult, calculate_stats
from .similarity import rouge_l


class RougeLDiversity(BaseEvaluator):
    """Diversity evaluator returning pairwise ROUGE-L distance scores."""

    instance_plot_metrics = [("pairwise_rouge_l_distance", "histogram")]
    aggregate_plot_metrics = ["pairwise_rouge_l_distance"]
    key_plot_metrics = [("pairwise_rouge_l_distance", "Pairwise ROUGE-L Distance")]

    def __init__(self, num_workers: int = 1):
        super().__init__(name="RougeLDiversity", num_workers=num_workers)

    def _extract_texts(self, input_data: dict[str, Any]) -> list[str]:
        values = None
        if "texts" in input_data:
            values = input_data["texts"]
        elif "items" in input_data:
            values = input_data["items"]
        else:
            raise KeyError("Expected 'texts' or 'items' in input_data.")

        texts = []
        for item in values:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "description" in item:
                texts.append(str(item["description"]))
            else:
                texts.append(str(item))
        return texts

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        texts = self._extract_texts(input_data)
        n = len(texts)
        if n < 2:
            return {"pairwise_distance_scores": [], "distance_matrix": np.array([])}

        matrix = np.zeros((n, n), dtype=float)
        scores: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                similarity = rouge_l(texts[i], texts[j])
                distance = float(1.0 - similarity)
                matrix[i, j] = distance
                matrix[j, i] = distance
                scores.append(distance)
        return {"pairwise_distance_scores": scores, "distance_matrix": matrix}

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        flat = [score for m in instance_metrics for score in m["pairwise_distance_scores"]]
        return {"pairwise_rouge_l_distance": calculate_stats(flat)}

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        if metadata is None:
            metadata = {}
        metadata.update({"evaluation_framework": "ROUGE-L Diversity"})
        return super().evaluate(inputs, metadata=metadata)
