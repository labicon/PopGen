from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def evenly_spaced_indices(total_items: int, max_items: int | None) -> list[int]:
    """Return up to ``max_items`` indices distributed across ``total_items``."""
    if total_items < 0:
        raise ValueError("total_items must be non-negative.")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive when provided.")
    if max_items is None or total_items <= max_items:
        return list(range(total_items))
    return [(((index + 1) * total_items) - 1) // max_items for index in range(max_items)]


def select_evenly_spaced_items[T](items: Sequence[T], max_items: int | None) -> list[T]:
    """Return up to ``max_items`` items at the end of equal-sized partitions."""
    return [items[index] for index in evenly_spaced_indices(len(items), max_items)]


def select_evenly_spaced_paths(paths: Sequence[str | Path], max_paths: int | None) -> list[Path]:
    """Return evenly distributed ``Path`` objects from a checkpoint list."""
    resolved_paths = [Path(path) for path in paths]
    return select_evenly_spaced_items(resolved_paths, max_paths)
