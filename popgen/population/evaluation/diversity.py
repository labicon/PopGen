"""Embedding-based diversity evaluators."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.stats import norm as sp_norm
from scipy.stats.qmc import Sobol

from popgen.population.encode import Encoder, TextEncoder
from popgen.utils.config import LitellmConfig

from .base import BaseEvaluator, EvalResult, calculate_stats
from .similarity import dot_product


@lru_cache(maxsize=32)
def _calibrate_radius(
    n_population: int,
    dim: int,
    n_reference_points: int,
    n_calibration_rounds: int,
    calibration_target: float,
    seed: int,
) -> float:
    """Calibrate coverage radius using Sobol quasi-random reference populations.

    For each round, generates a synthetic population of size *n_population* on
    the unit hypersphere via a scrambled Sobol sequence, then finds the smallest
    Euclidean radius k such that *calibration_target* fraction of the reference
    points are within distance k of at least one synthetic point.  The final
    radius is the average across all rounds.
    """
    rng = np.random.default_rng(seed)

    # Reference points on the unit sphere (reused across rounds)
    ref = rng.normal(size=(n_reference_points, dim))
    ref /= np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)

    radii = np.empty(n_calibration_rounds)
    for i in range(n_calibration_rounds):
        sobol = Sobol(d=dim, scramble=True, seed=seed + i + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # non-power-of-2 n
            pts = sobol.random(n_population)  # (n_population, dim) in [0, 1]^dim
        pts = np.clip(pts, 1e-6, 1.0 - 1e-6)
        gauss = sp_norm.ppf(pts)
        gauss /= np.clip(np.linalg.norm(gauss, axis=1, keepdims=True), 1e-12, None)

        # Euclidean distances for L2-normalised vectors: d = sqrt(2(1 - cos))
        sims = ref @ gauss.T  # (n_reference_points, n_population)
        dists = np.sqrt(np.clip(2.0 * (1.0 - np.clip(sims, -1.0, 1.0)), 0.0, None))
        min_dists = np.min(dists, axis=1)

        radii[i] = float(np.percentile(min_dists, calibration_target * 100))

    return float(np.mean(radii))


class EmbeddingDiversityEvaluator(BaseEvaluator):
    """Diversity evaluator over embedding geometry."""

    instance_plot_metrics = [
        ("average_pairwise_distance", "histogram"),
        ("minimum_pairwise_distance", "histogram"),
        ("coverage", "histogram"),
        ("dispersion", "histogram"),
    ]
    aggregate_plot_metrics = [
        "average_pairwise_distance",
        "minimum_pairwise_distance",
        "convex_hull_volume",
        "coverage",
        "dispersion",
    ]
    key_plot_metrics = [("average_pairwise_distance", "Average Pairwise Distance")]

    def __init__(
        self,
        encoder: Encoder,
        num_workers: int = 1,
        dot_product_norm: bool = True,
        coverage_samples: int = 1_000,
        calibration_rounds: int = 1_000,
        calibration_target: float = 0.99,
        monte_carlo_seed: int = 0,
        volume_eps: float = 1e-8,
        save_path: str | None = None,
    ):
        super().__init__(name="EmbeddingDiversity", num_workers=num_workers)
        self.encoder = encoder
        self.dot_product_norm = bool(dot_product_norm)
        self.coverage_samples = int(coverage_samples)
        self.calibration_rounds = int(calibration_rounds)
        self.calibration_target = float(calibration_target)
        self.monte_carlo_seed = int(monte_carlo_seed)
        self.volume_eps = float(volume_eps)
        self.save_path = save_path

    def _extract_items(self, input_data: dict[str, Any]) -> list[Any]:
        if "items" in input_data:
            return list(input_data["items"])
        if "texts" in input_data:
            texts = []
            for item in input_data.get("texts", []):
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "description" in item:
                    texts.append(item["description"])
                else:
                    texts.append(str(item))
            return texts
        if "trajectories" in input_data:
            return list(input_data["trajectories"])
        raise KeyError("Expected one of: 'items', 'texts', or 'trajectories'.")

    def _pairwise_distances_from_similarity(self, similarity_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(similarity_matrix, -1.0, 1.0)
        distance_matrix = np.sqrt(np.clip(2.0 * (1.0 - clipped), 0.0, None))
        tri = np.triu_indices(distance_matrix.shape[0], k=1)
        return distance_matrix, distance_matrix[tri]

    def _convex_hull_volume_proxy(self, similarity_matrix: np.ndarray) -> float:
        gram = 0.5 * (similarity_matrix + similarity_matrix.T)
        gram = gram + np.eye(gram.shape[0]) * self.volume_eps
        sign, logdet = np.linalg.slogdet(gram)
        if sign <= 0:
            return 0.0
        return float(np.exp(0.5 * logdet))

    def _coverage_and_dispersion(self, embeddings: np.ndarray) -> tuple[float, float]:
        n, dim = embeddings.shape
        if n == 0 or dim == 0:
            return 0.0, 0.0

        rng = np.random.default_rng(self.monte_carlo_seed)
        samples = rng.normal(size=(self.coverage_samples, dim))
        samples /= np.clip(np.linalg.norm(samples, axis=1, keepdims=True), 1e-12, None)

        # Euclidean distances for L2-normalised vectors: d = sqrt(2(1 - cos))
        sims = dot_product(samples, embeddings, norm=self.dot_product_norm)
        dists = np.sqrt(np.clip(2.0 * (1.0 - np.clip(sims, -1.0, 1.0)), 0.0, None))
        min_dists = np.min(dists, axis=1)

        # Adaptive radius calibrated via Sobol reference populations
        k = _calibrate_radius(
            n_population=n,
            dim=dim,
            n_reference_points=10_000,
            n_calibration_rounds=self.calibration_rounds,
            calibration_target=self.calibration_target,
            seed=self.monte_carlo_seed,
        )

        coverage = float(np.mean(min_dists <= k))
        dispersion = float(np.max(min_dists))
        return coverage, dispersion

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        items = self._extract_items(input_data)
        n_items = len(items)
        if n_items < 2:
            return {
                "num_items": n_items,
                "similarity_matrix": np.array([]),
                "distance_matrix": np.array([]),
                "pairwise_distances": [],
                "average_pairwise_distance": 0.0,
                "minimum_pairwise_distance": 0.0,
                "convex_hull_volume": 0.0,
                "coverage": 0.0,
                "dispersion": 0.0,
                "embeddings": np.array([]),
            }

        embeddings = np.asarray(self.encoder.batch_encoder(items), dtype=float)
        if embeddings.ndim != 2:
            raise ValueError(f"Encoder output must be 2D, got shape {embeddings.shape}.")

        similarity_matrix = dot_product(embeddings, norm=self.dot_product_norm)
        distance_matrix, pairwise_distances = self._pairwise_distances_from_similarity(similarity_matrix)

        average_pairwise_distance = float(np.mean(pairwise_distances)) if pairwise_distances.size > 0 else 0.0
        minimum_pairwise_distance = float(np.min(pairwise_distances)) if pairwise_distances.size > 0 else 0.0
        convex_hull_volume = self._convex_hull_volume_proxy(similarity_matrix)
        coverage, dispersion = self._coverage_and_dispersion(embeddings)

        return {
            "num_items": n_items,
            "similarity_matrix": similarity_matrix,
            "distance_matrix": distance_matrix,
            "pairwise_distances": pairwise_distances.tolist(),
            "average_pairwise_distance": average_pairwise_distance,
            "minimum_pairwise_distance": minimum_pairwise_distance,
            "convex_hull_volume": convex_hull_volume,
            "coverage": coverage,
            "dispersion": dispersion,
            "embeddings": embeddings,
        }

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if self.save_path is not None:
            for i, m in enumerate(instance_metrics):
                if "embeddings" in m:
                    np.save(f"{self.save_path}/embeddings_{i}.npy", np.asarray(m["embeddings"]))

        return {
            "average_pairwise_distance": calculate_stats([m["average_pairwise_distance"] for m in instance_metrics]),
            "minimum_pairwise_distance": calculate_stats([m["minimum_pairwise_distance"] for m in instance_metrics]),
            "convex_hull_volume": calculate_stats([m["convex_hull_volume"] for m in instance_metrics]),
            "coverage": calculate_stats([m["coverage"] for m in instance_metrics]),
            "dispersion": calculate_stats([m["dispersion"] for m in instance_metrics]),
        }

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "Embedding Diversity (Dot-Product Geometry)",
                "encoder": type(self.encoder).__name__,
                "dot_product_norm": self.dot_product_norm,
                "coverage_samples": self.coverage_samples,
                "calibration_rounds": self.calibration_rounds,
                "calibration_target": self.calibration_target,
            }
        )

        result = super().evaluate(inputs, metadata=metadata)

        for m in result.instance_metrics:
            if "embeddings" in m:
                del m["embeddings"]

        return result


class DiversityEvaluator(EmbeddingDiversityEvaluator):
    """Backward-compatible text diversity wrapper."""

    def __init__(
        self,
        embedding_model: str | LitellmConfig,
        num_workers: int = 1,
        batch_size: int = 32,
        save_path: str | None = None,
        **kwargs,
    ):
        encoder = TextEncoder(embedding_model=embedding_model, batch_size=batch_size)
        super().__init__(
            encoder=encoder,
            num_workers=num_workers,
            save_path=save_path,
            **kwargs,
        )
