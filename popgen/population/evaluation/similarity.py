"""Similarity functions for text, embeddings, and trajectories."""

from __future__ import annotations

from typing import Any

import litellm
import numpy as np
from rouge_score import rouge_scorer


def _as_2d(array_like: Any) -> np.ndarray:
    arr = np.asarray(array_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array-like input, got shape {arr.shape}.")
    return arr


def dot_product(
    embeddings1: Any,
    embeddings2: Any | None = None,
    norm: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute pairwise dot-product similarity matrix."""
    e1 = _as_2d(embeddings1)
    e2 = e1 if embeddings2 is None else _as_2d(embeddings2)

    if norm:
        e1 = e1 / np.clip(np.linalg.norm(e1, axis=1, keepdims=True), eps, None)
        e2 = e2 / np.clip(np.linalg.norm(e2, axis=1, keepdims=True), eps, None)

    return e1 @ e2.T


def _flatten_numeric(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return _flatten_numeric(value.tolist())
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
            out.extend(_flatten_numeric(item))
        return out
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    return []


def _step_to_vector(step: Any) -> np.ndarray:
    if isinstance(step, dict) and "state" in step:
        step = step["state"]
    flat = _flatten_numeric(step)
    if not flat:
        return np.zeros(1, dtype=float)
    return np.asarray(flat, dtype=float)


def _sequence_to_vectors(trajectory: Any) -> list[np.ndarray]:
    if trajectory is None:
        return []
    if isinstance(trajectory, np.ndarray):
        if trajectory.ndim == 1:
            return [trajectory.astype(float)]
        return [np.asarray(row, dtype=float).reshape(-1) for row in trajectory]
    if not isinstance(trajectory, list):
        trajectory = [trajectory]
    return [_step_to_vector(step) for step in trajectory]


def _step_distance(v1: np.ndarray, v2: np.ndarray, distance_metric: str) -> float:
    max_dim = max(v1.size, v2.size)
    a = np.pad(v1, (0, max_dim - v1.size), constant_values=0.0)
    b = np.pad(v2, (0, max_dim - v2.size), constant_values=0.0)
    if distance_metric == "manhattan":
        return float(np.abs(a - b).sum())
    if distance_metric != "euclidean":
        raise ValueError(f"Unsupported distance_metric '{distance_metric}'.")
    return float(np.linalg.norm(a - b))


def dtw(
    trajectory1: Any,
    trajectory2: Any,
    distance_metric: str = "euclidean",
    normalized: bool = True,
) -> float:
    """Compute DTW similarity between trajectories."""
    seq1 = _sequence_to_vectors(trajectory1)
    seq2 = _sequence_to_vectors(trajectory2)
    n = len(seq1)
    m = len(seq2)

    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        non_empty = seq2 if n == 0 else seq1
        total = float(sum(np.linalg.norm(v) for v in non_empty))
        path_len = len(non_empty)
        dist = total / path_len if normalized and path_len > 0 else total
        return float(1.0 / (1.0 + dist))

    table = np.full((n + 1, m + 1), np.inf)
    path_len = np.zeros((n + 1, m + 1), dtype=np.int32)
    table[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _step_distance(seq1[i - 1], seq2[j - 1], distance_metric=distance_metric)
            prev = (table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
            k = int(np.argmin(prev))
            if k == 0:
                best_cost = prev[0]
                best_path = path_len[i - 1, j]
            elif k == 1:
                best_cost = prev[1]
                best_path = path_len[i, j - 1]
            else:
                best_cost = prev[2]
                best_path = path_len[i - 1, j - 1]
            table[i, j] = cost + best_cost
            path_len[i, j] = best_path + 1

    dist = float(table[n, m])
    if normalized and path_len[n, m] > 0:
        dist /= float(path_len[n, m])
    return float(1.0 / (1.0 + dist))


_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(text1: str, text2: str) -> float:
    """Compute ROUGE-L F1 similarity between two texts."""
    score = _ROUGE_SCORER.score(str(text1), str(text2))
    return float(score["rougeL"].fmeasure)
