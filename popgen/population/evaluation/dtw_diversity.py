"""DTW-based diversity evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEvaluator, EvalResult, calculate_stats
from .similarity import dtw


class DTWDiversity(BaseEvaluator):
    """Diversity evaluator returning pairwise DTW distance scores."""

    instance_plot_metrics = [("pairwise_dtw_distance", "histogram")]
    aggregate_plot_metrics = ["pairwise_dtw_distance"]
    key_plot_metrics = [("pairwise_dtw_distance", "Pairwise DTW Distance")]

    def __init__(self, num_workers: int = 1, distance_metric: str = "euclidean"):
        super().__init__(name="DTWDiversity", num_workers=num_workers)
        self.distance_metric = distance_metric

    def _extract_trajectories(self, input_data: dict[str, Any]) -> list[Any]:
        if "trajectories" in input_data:
            return list(input_data["trajectories"])
        if "items" in input_data:
            return list(input_data["items"])
        raise KeyError("Expected 'trajectories' or 'items' in input_data.")

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        trajectories = self._extract_trajectories(input_data)
        n = len(trajectories)
        if n < 2:
            return {"pairwise_distance_scores": [], "distance_matrix": np.array([])}

        matrix = np.zeros((n, n), dtype=float)
        scores: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                similarity = dtw(
                    trajectories[i],
                    trajectories[j],
                    distance_metric=self.distance_metric,
                    normalized=True,
                )
                distance = float(1.0 - similarity)
                matrix[i, j] = distance
                matrix[j, i] = distance
                scores.append(distance)
        return {"pairwise_distance_scores": scores, "distance_matrix": matrix}

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        flat = [score for m in instance_metrics for score in m["pairwise_distance_scores"]]
        return {"pairwise_dtw_distance": calculate_stats(flat)}

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        if metadata is None:
            metadata = {}
        metadata.update(
            {
                "evaluation_framework": "DTW Diversity",
                "distance_metric": self.distance_metric,
            }
        )
        return super().evaluate(inputs, metadata=metadata)
