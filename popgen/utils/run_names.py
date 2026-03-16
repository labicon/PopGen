"""Helpers for building deterministic training run names."""

from __future__ import annotations

import re
from pathlib import Path

_RUN_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?$"),
    re.compile(r"^\d{8}_\d{6}$"),
)


def extract_run_date_from_path(path: str | Path) -> str:
    """Extract run-date token from path, falling back to basename."""
    path_obj = Path(path)
    for part in reversed(path_obj.parts):
        if any(pattern.fullmatch(part) for pattern in _RUN_DATE_PATTERNS):
            return part
    return path_obj.name


def build_cooperator_run_name(env_name: str, run_date: str) -> str:
    """Build canonical cooperator-training run name."""
    return f"{env_name}_{run_date}_cooperator"
