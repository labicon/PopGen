"""Base evaluator class for behavior evaluation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from pqdm.threads import pqdm


@dataclass
class EvalResult:
    """Container for evaluation results."""

    instance_metrics: list[dict[str, Any]]
    overall_metrics: dict[str, Any]
    metadata: dict[str, Any] | None = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_metrics": self.instance_metrics,
            "overall_metrics": self.overall_metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        """Create from dictionary."""
        return cls(
            instance_metrics=data["instance_metrics"],
            overall_metrics=data["overall_metrics"],
            metadata=data.get("metadata", {}),
        )


class EvalResultEncoder(json.JSONEncoder):
    """JSON encoder for EvalResult that handles numpy types."""

    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    instance_plot_metrics: list[tuple[str, str]] = []
    aggregate_plot_metrics: list[str] = []
    key_plot_metrics: list[tuple[str, str]] = []

    def __init__(self, name: str, num_workers: int = 8):
        self.name = name
        self.num_workers = num_workers

    @abstractmethod
    def compute_instance_metric(self, input_data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Compute metrics for a single instance."""

    @abstractmethod
    def aggregate_metrics(self, instance_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate instance-level metrics into overall metrics."""

    def evaluate(
        self,
        inputs: list[Any],
        metadata: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate a list of inputs.

        Args:
            inputs: List of inputs to evaluate (format depends on subclass)
            metadata: Additional metadata about the evaluation

        Returns:
            EvalResult containing instance and overall metrics
        """
        instance_metrics = pqdm(
            inputs,
            self.compute_instance_metric,
            n_jobs=self.num_workers,
            desc=f"Computing {self.name} metrics",
        )

        overall_metrics = self.aggregate_metrics(instance_metrics)
        return EvalResult(instance_metrics, overall_metrics, metadata)

    def save_results(self, result: EvalResult, output_path: str | Path):
        """Save evaluation results to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=4, cls=EvalResultEncoder)

    @classmethod
    def load_results(cls, input_path: str | Path) -> EvalResult:
        """Load evaluation results from a file."""
        with open(input_path) as f:
            data = json.load(f)
        return EvalResult.from_dict(data)


def calculate_stats(values: list[float], num_responses_per_prompt: int | None = None) -> dict[str, float]:
    """
    Calculate basic statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary containing mean, std, min, max, etc.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}

    values = np.array([float(v) for v in values if v is not None and not np.isnan(v)])

    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}

    return {
        "mean": (float(np.sum(values) / num_responses_per_prompt) if num_responses_per_prompt else float(np.mean(values))),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "count": len(values),
    }
