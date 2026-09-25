"""Helpers for building deterministic training run names."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_RUN_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?$"),
    re.compile(r"^\d{8}_\d{6}$"),
)
_BEHAVIOR_RE = re.compile(r"^behavior_(\d+)$")
_ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")


def infer_env_name(module_or_module_name: Any) -> str:
    """Infer environment name from module object/string."""
    if isinstance(module_or_module_name, str):
        module_name = module_or_module_name
    elif hasattr(module_or_module_name, "__name__"):
        module_name = str(module_or_module_name.__name__)
    else:
        raise TypeError("Expected module object or module-name string for env-name inference.")

    if module_name and "." not in module_name:
        return module_name

    parts = module_name.split(".")
    if "envs" in parts:
        idx = parts.index("envs")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    raise ValueError(f"Could not infer environment name from '{module_name}'.")


def extract_run_date_from_path(path: str | Path) -> str:
    """Extract run-date token from path, falling back to basename."""
    path_obj = Path(path)
    for part in reversed(path_obj.parts):
        if any(pattern.fullmatch(part) for pattern in _RUN_DATE_PATTERNS):
            return part
    return path_obj.name


def extract_behavior_idx_from_path(path: str | Path) -> int:
    """Extract behavior index from a path segment like ``behavior_3``."""
    for part in reversed(Path(path).parts):
        match = _BEHAVIOR_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not find behavior_<n> segment in path '{path}'.")


def extract_attempt_idx_from_path(path: str | Path) -> int:
    """Extract attempt index from a path segment like ``attempt_0``."""
    for part in reversed(Path(path).parts):
        match = _ATTEMPT_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not find attempt_<n> segment in path '{path}'.")


def build_population_run_name(
    env_name: str,
    run_date: str,
    behavior_idx: int,
    attempt_idx: int,
) -> str:
    """Build canonical population-training run name."""
    return f"{env_name}_{run_date}_behavior_{int(behavior_idx)}_attempt_{int(attempt_idx)}"


def build_cooperator_run_name(env_name: str, run_date: str) -> str:
    """Build canonical cooperator-training run name."""
    return f"{env_name}_{run_date}_cooperator"
