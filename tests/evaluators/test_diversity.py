import numpy as np

from popgen.population.encode import Encoder
from popgen.population.evaluation import (
    DTWDiversity,
    EmbeddingDiversityEvaluator,
    RougeLDiversity,
)


class _IdentityEncoder(Encoder):
    def encode(self, input):
        return np.asarray(input, dtype=float)

    def batch_encoder(self, inputs):
        return np.asarray(inputs, dtype=float)


def test_embedding_diversity_pairwise_metrics_identity_embeddings():
    evaluator = EmbeddingDiversityEvaluator(
        encoder=_IdentityEncoder(),
        dot_product_norm=True,
        coverage_samples=128,
        calibration_rounds=10,
        monte_carlo_seed=0,
    )
    metrics = evaluator.compute_instance_metric(
        {"items": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
    )

    expected = float(np.sqrt(2.0))
    assert metrics["average_pairwise_distance"] == expected
    assert metrics["minimum_pairwise_distance"] == expected
    assert metrics["convex_hull_volume"] > 0.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["dispersion"] >= 0.0
    assert len(metrics["pairwise_distances"]) == 3


def test_embedding_diversity_single_item_returns_zeros():
    evaluator = EmbeddingDiversityEvaluator(encoder=_IdentityEncoder())
    metrics = evaluator.compute_instance_metric({"items": [[1.0, 0.0]]})
    assert metrics["average_pairwise_distance"] == 0.0
    assert metrics["minimum_pairwise_distance"] == 0.0
    assert metrics["coverage"] == 0.0
    assert metrics["dispersion"] == 0.0


def test_dtw_diversity_returns_pairwise_scores():
    evaluator = DTWDiversity(num_workers=1)
    metrics = evaluator.compute_instance_metric(
        {
            "trajectories": [
                [[0.0], [1.0], [2.0]],
                [[0.0], [1.0], [2.0]],
                [[2.0], [2.0], [2.0]],
            ]
        }
    )
    assert len(metrics["pairwise_distance_scores"]) == 3
    assert metrics["distance_matrix"].shape == (3, 3)


def test_rouge_l_diversity_returns_pairwise_scores():
    evaluator = RougeLDiversity(num_workers=1)
    metrics = evaluator.compute_instance_metric(
        {"texts": ["go to goal", "go to goal", "wait and observe"]}
    )
    assert len(metrics["pairwise_distance_scores"]) == 3
    assert metrics["distance_matrix"].shape == (3, 3)
