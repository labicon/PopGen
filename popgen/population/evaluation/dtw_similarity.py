"""DTW-based trajectory similarity evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEvaluator, EvalResult, calculate_stats


class DTWSimilarityEvaluator(BaseEvaluator):
    """Evaluator for trajectory similarity using Dynamic Time Warping (DTW)."""

    instance_plot_metrics = [
        ("dtw_similarity", "histogram"),
        ("dtw_distance", "histogram"),
    ]
    aggregate_plot_metrics = ["dtw_similarity", "dtw_distance", "normalized_dtw_distance"]
    key_plot_metrics = [("dtw_similarity", "DTW Similarity")]

    def __init__(
        self,
        num_workers: int = 8,
        agent_name: str = "ego_agent",
        group_name: str | None = "ego_team",
        feature_key: str = "state",
        distance_metric: str = "euclidean",
    ):
        super().__init__(name="DTWSimilarity", num_workers=num_workers)
        if distance_metric not in {"euclidean", "manhattan"}:
            raise ValueError(f"Unsupported distance_metric '{distance_metric}'.")
        self.agent_name = agent_name
        self.group_name = group_name
        self.feature_key = feature_key
        self.distance_metric = distance_metric

    def _flatten_numeric(self, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return self._flatten_numeric(value.tolist())
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, bool):
            return [float(value)]
        if isinstance(value, (int, float)):
            if np.isnan(value):
                return []
            return [float(value)]
        if isinstance(value, dict):
            out: list[float] = []
            for item in value.values():
                out.extend(self._flatten_numeric(item))
            return out
        if isinstance(value, (list, tuple)):
            out: list[float] = []
            for item in value:
                out.extend(self._flatten_numeric(item))
            return out
        return []

    def _extract_agent_feature(self, step: dict[str, Any], agent_name: str, group_name: str | None, feature_key: str) -> Any | None:
        candidate_agent_names = [agent_name]
        if agent_name == "ego_agent":
            candidate_agent_names.append("agent0")
        candidate_group_names: list[str] = []
        if group_name is not None:
            candidate_group_names.append(group_name)
            if group_name == "ego_team":
                candidate_group_names.extend(["ego", "agents", "agent_blue"])
            elif group_name == "opponent_team":
                candidate_group_names.extend(["opponent", "opponents", "agent_red"])

        agents = step.get("agents")
        if isinstance(agents, dict):
            for gname in candidate_group_names:
                if isinstance(agents.get(gname), dict):
                    for aname in candidate_agent_names:
                        payload = agents[gname].get(aname)
                        if isinstance(payload, dict) and feature_key in payload:
                            return payload[feature_key]

            for aname in candidate_agent_names:
                payload = agents.get(aname)
                if isinstance(payload, dict) and feature_key in payload:
                    return payload[feature_key]

            for maybe_group in agents.values():
                if not isinstance(maybe_group, dict):
                    continue
                for aname in candidate_agent_names:
                    payload = maybe_group.get(aname)
                    if isinstance(payload, dict) and feature_key in payload:
                        return payload[feature_key]

        for gname in candidate_group_names:
            if isinstance(step.get(gname), dict):
                for aname in candidate_agent_names:
                    payload = step[gname].get(aname)
                    if isinstance(payload, dict) and feature_key in payload:
                        return payload[feature_key]

        for key, value in step.items():
            if key in {"step", "env"} or not isinstance(value, dict):
                continue
            for aname in candidate_agent_names:
                payload = value.get(aname)
                if isinstance(payload, dict) and feature_key in payload:
                    return payload[feature_key]

        return None

    def _step_to_vector(self, step: Any, agent_name: str, group_name: str | None, feature_key: str) -> np.ndarray:
        selected: Any = step
        if isinstance(step, dict):
            extracted = self._extract_agent_feature(step, agent_name=agent_name, group_name=group_name, feature_key=feature_key)
            if extracted is not None:
                selected = extracted
            elif feature_key in step:
                selected = step[feature_key]

        flattened = self._flatten_numeric(selected)
        if not flattened:
            return np.zeros(1, dtype=float)
        return np.asarray(flattened, dtype=float)

    def _as_trajectory_list(self, trajectory: Any) -> list[Any]:
        if trajectory is None:
            return []
        if isinstance(trajectory, list):
            return trajectory
        if isinstance(trajectory, tuple):
            return list(trajectory)
        if isinstance(trajectory, np.ndarray):
            return trajectory.tolist()
        return [trajectory]

    def _extract_trajectory(self, trajectory: list[Any], agent_name: str, group_name: str | None, feature_key: str) -> list[np.ndarray]:
        return [self._step_to_vector(step, agent_name=agent_name, group_name=group_name, feature_key=feature_key) for step in trajectory]

    def _step_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        max_dim = max(v1.size, v2.size)
        a = np.pad(v1, (0, max_dim - v1.size), constant_values=0.0)
        b = np.pad(v2, (0, max_dim - v2.size), constant_values=0.0)

        if self.distance_metric == "manhattan":
            return float(np.abs(a - b).sum())
        return float(np.linalg.norm(a - b))

    def _dtw_distance(self, seq1: list[np.ndarray], seq2: list[np.ndarray]) -> tuple[float, int]:
        n = len(seq1)
        m = len(seq2)

        if n == 0 and m == 0:
            return 0.0, 0
        if n == 0:
            total = float(sum(np.linalg.norm(v) for v in seq2))
            return total, len(seq2)
        if m == 0:
            total = float(sum(np.linalg.norm(v) for v in seq1))
            return total, len(seq1)

        dtw = np.full((n + 1, m + 1), np.inf)
        path_len = np.zeros((n + 1, m + 1), dtype=np.int32)
        dtw[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._step_distance(seq1[i - 1], seq2[j - 1])
                prev_costs = (dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
                prev_idx = int(np.argmin(prev_costs))
                if prev_idx == 0:
                    best_prev_cost = prev_costs[0]
                    best_prev_path = path_len[i - 1, j]
                elif prev_idx == 1:
                    best_prev_cost = prev_costs[1]
                    best_prev_path = path_len[i, j - 1]
                else:
                    best_prev_cost = prev_costs[2]
                    best_prev_path = path_len[i - 1, j - 1]
                dtw[i, j] = cost + best_prev_cost
                path_len[i, j] = best_prev_path + 1

        return float(dtw[n, m]), int(path_len[n, m])

    def compute_instance_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Compute DTW similarity for one trajectory pair."""
        trajectory1 = input_data.get("trajectory1", input_data.get("rollout_states1"))
        trajectory2 = input_data.get("trajectory2", input_data.get("rollout_states2"))
        if trajectory1 is None or trajectory2 is None:
            raise KeyError("Input must include 'trajectory1'/'trajectory2' (or 'rollout_states1'/'rollout_states2').")

        agent_name = input_data.get("agent_name", self.agent_name)
        group_name = input_data.get("group_name", self.group_name)
        feature_key = input_data.get("feature_key", self.feature_key)

        raw_trajectory1 = self._as_trajectory_list(trajectory1)
        raw_trajectory2 = self._as_trajectory_list(trajectory2)
        seq1 = self._extract_trajectory(raw_trajectory1, agent_name=agent_name, group_name=group_name, feature_key=feature_key)
        seq2 = self._extract_trajectory(raw_trajectory2, agent_name=agent_name, group_name=group_name, feature_key=feature_key)

        dtw_distance, dtw_path_length = self._dtw_distance(seq1, seq2)
        normalized_distance = dtw_distance / dtw_path_length if dtw_path_length > 0 else 0.0
        dtw_similarity = 1.0 / (1.0 + normalized_distance)

        return {
            "agent_name": agent_name,
            "group_name": group_name,
            "feature_key": feature_key,
            "trajectory1_length": len(raw_trajectory1),
            "trajectory2_length": len(raw_trajectory2),
            "dtw_path_length": dtw_path_length,
            "dtw_distance": float(dtw_distance),
            "normalized_dtw_distance": float(normalized_distance),
            "dtw_similarity": float(dtw_similarity),
        }

    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate DTW metrics across instances."""
        return {
            "dtw_similarity": calculate_stats([m["dtw_similarity"] for m in instance_metrics]),
            "dtw_distance": calculate_stats([m["dtw_distance"] for m in instance_metrics]),
            "normalized_dtw_distance": calculate_stats([m["normalized_dtw_distance"] for m in instance_metrics]),
        }

    def evaluate(self, inputs: list[Any], metadata: dict[str, Any] = None) -> EvalResult:
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_framework": "DTW Trajectory Similarity",
                "distance_metric": self.distance_metric,
                "default_agent_name": self.agent_name,
                "default_group_name": self.group_name,
                "default_feature_key": self.feature_key,
            }
        )

        return super().evaluate(inputs, metadata=metadata)
