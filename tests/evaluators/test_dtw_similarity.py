import pytest

from popgen.population import DTWSimilarityEvaluator


def test_dtw_similarity_identical_trajectories():
    evaluator = DTWSimilarityEvaluator(num_workers=1)
    metrics = evaluator.compute_instance_metric(
        {
            "trajectory1": [[0.0], [1.0], [2.0]],
            "trajectory2": [[0.0], [1.0], [2.0]],
        }
    )

    assert metrics["dtw_distance"] == 0.0
    assert metrics["normalized_dtw_distance"] == 0.0
    assert metrics["dtw_similarity"] == 1.0
    assert metrics["dtw_path_length"] > 0


def test_dtw_similarity_non_identical_trajectories():
    evaluator = DTWSimilarityEvaluator(num_workers=1)
    metrics = evaluator.compute_instance_metric(
        {
            "trajectory1": [[0.0], [1.0], [2.0]],
            "trajectory2": [[0.0], [2.0]],
        }
    )

    assert metrics["dtw_distance"] > 0.0
    assert metrics["normalized_dtw_distance"] > 0.0
    assert 0.0 < metrics["dtw_similarity"] < 1.0


def test_dtw_similarity_extracts_agent_state_from_grouped_rollout_entries():
    evaluator = DTWSimilarityEvaluator(num_workers=1, agent_name="agent0", group_name="ego", feature_key="state")

    trajectory1 = [
        {"ego": {"agent0": {"state": [0.0, 0.0], "action": 0, "reward": 0.0}}},
        {"ego": {"agent0": {"state": [1.0, 1.0], "action": 1, "reward": 0.0}}},
    ]
    trajectory2 = [
        {"ego": {"agent0": {"state": [0.0, 0.0], "action": 0, "reward": 0.0}}},
        {"ego": {"agent0": {"state": [1.2, 1.1], "action": 1, "reward": 0.0}}},
    ]

    metrics = evaluator.compute_instance_metric({"trajectory1": trajectory1, "trajectory2": trajectory2})
    assert metrics["trajectory1_length"] == 2
    assert metrics["trajectory2_length"] == 2
    assert metrics["dtw_distance"] > 0.0
    assert metrics["dtw_similarity"] < 1.0


def test_dtw_similarity_evaluate_sets_metadata():
    evaluator = DTWSimilarityEvaluator(num_workers=1)
    result = evaluator.evaluate(
        [
            {
                "trajectory1": [[0.0], [1.0], [2.0]],
                "trajectory2": [[0.0], [1.0], [2.0]],
            },
            {
                "trajectory1": [[0.0], [1.0], [2.0]],
                "trajectory2": [[0.0], [2.0]],
            },
        ]
    )

    assert "dtw_similarity" in result.overall_metrics
    assert result.metadata["evaluation_framework"] == "DTW Trajectory Similarity"
    assert result.metadata["distance_metric"] == "euclidean"


def test_dtw_similarity_requires_trajectory_inputs():
    evaluator = DTWSimilarityEvaluator(num_workers=1)
    with pytest.raises(KeyError):
        evaluator.compute_instance_metric({"trajectory1": [[0.0], [1.0]]})
