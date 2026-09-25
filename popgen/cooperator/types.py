"""Type definitions for cooperator training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyArtifactRef:
    """Reference to a teammate policy artifact.

    Attributes:
        kind: Artifact type identifier (``"PolicyCode"`` or ``"RewardFn"``).
            This determines which policy loader is expected to parse the file.
        path: Absolute or relative path to the artifact payload.
        metadata: Optional source-specific metadata used for logging/debugging
            (for example adherence score or attempt index).
    """

    kind: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
