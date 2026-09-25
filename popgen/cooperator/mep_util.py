"""Enumerate the members of a MEP population as teammate policy references.

A MEP run writes ``mep_population/population_manifest.json`` listing one entry per
population member, each pointing at a BenchMARL run directory. Those members are
ordinary ``RewardFn`` artifacts -- ``popgen.cooperator.util.load_policies`` already
knows how to reload one and wrap it for the tensor-only teammate API -- so this module
holds only the parts that are genuinely MEP-specific: locating the manifest, resolving
member roots that were recorded on another machine, and skipping members that never
produced a checkpoint.

Used by :class:`~popgen.cooperator.evaluation.CooperatorEvaluator` with
``eval_type="mep_population"`` to evaluate any method's cooperator against a MEP
population as a held-out partner set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from popgen.cooperator.types import PolicyArtifactRef
from popgen.utils.filesystem import find_reward_checkpoints

MANIFEST_FILENAME = "population_manifest.json"


def resolve_manifest_path(population_path: str | Path) -> Path:
    """Accept either the ``mep_population/`` directory or the manifest file itself."""
    path = Path(population_path).expanduser().resolve()
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No MEP population manifest at {path}")
    return path


def _resolve_member_root(member_root: str | Path, manifest_path: Path) -> Path:
    """Locate a member's run directory, re-anchoring paths from another machine.

    Manifests record absolute paths, so one written elsewhere -- or before the checkout
    moved -- points nowhere and every member is silently skipped. Members always live
    under the manifest's own directory, so re-anchor the recorded path there before
    giving up: that keeps a copied results tree evaluable without hand-editing its JSON.

    Kept behaviourally identical to ``scripts/mep/eval_common._resolve_member_root``.
    """
    path = Path(member_root)
    if not path.is_absolute():
        return (manifest_path.parent / path).resolve()

    resolved = path.resolve()
    if resolved.exists():
        return resolved

    parts = resolved.parts
    if "members" in parts:
        relocated = (manifest_path.parent / Path(*parts[parts.index("members") :])).resolve()
        if relocated.exists():
            logging.warning(
                "Re-anchoring MEP member root %s to %s because the recorded path is not present on this machine.",
                resolved,
                relocated,
            )
            return relocated

    return resolved


def _latest_checkpoint(member_root: Path) -> Path | None:
    """Newest checkpoint under *member_root*, by the frame count in the filename.

    ``find_reward_checkpoints`` orders by mtime, which a copy without ``-t`` scrambles.
    The frame count is intrinsic to the artifact, so prefer it and fall back to the
    mtime order only for names that do not carry one.
    """
    checkpoints = find_reward_checkpoints(member_root)
    if not checkpoints:
        return None

    def sort_key(path: Path) -> tuple[int, int]:
        stem = path.stem
        _, _, suffix = stem.partition("checkpoint")
        digits = suffix.lstrip("_")
        return (1, int(digits)) if digits.isdigit() else (0, 0)

    return sorted(checkpoints, key=sort_key)[-1]


def enumerate_mep_member_refs(
    population_path: str | Path,
    *,
    expected_env: str | None = None,
) -> list[PolicyArtifactRef]:
    """Return one ``RewardFn`` ref per loadable MEP population member, by ``member_id``.

    Args:
        population_path: The ``mep_population/`` directory or its manifest file.
        expected_env: If given, the manifest's ``env`` must match. ``pp`` and ``pp_stag``
            both register the VMAS scenario ``"simple_tag"`` with different observation
            widths, so loading a population under the wrong env module rebuilds a
            mismatched model and fails deep inside the checkpoint reload. Checking here
            turns that into an immediate, legible error.

    Raises:
        ValueError: if *expected_env* does not match, or no member has a checkpoint.
    """
    manifest_path = resolve_manifest_path(population_path)
    with open(manifest_path) as f:
        manifest: dict[str, Any] = json.load(f)

    manifest_env = str(manifest.get("env", "") or "").strip()
    if expected_env is not None and manifest_env and manifest_env != expected_env:
        raise ValueError(
            f"MEP population at {manifest_path} was trained on env {manifest_env!r}, "
            f"but this evaluation runs env {expected_env!r}. Loading a population under "
            "the wrong env module rebuilds a mismatched model."
        )

    refs: list[PolicyArtifactRef] = []
    skipped: list[str] = []
    for idx, member in enumerate(manifest.get("members", [])):
        member_root_raw = member.get("member_root")
        if not member_root_raw:
            continue
        member_id = int(member.get("member_id", idx))
        member_root = _resolve_member_root(member_root_raw, manifest_path)
        if _latest_checkpoint(member_root) is None:
            skipped.append(f"member_{member_id} ({member_root})")
            continue
        refs.append(
            PolicyArtifactRef(
                kind="RewardFn",
                path=member_root,
                metadata={
                    "member_id": member_id,
                    "manifest_path": str(manifest_path),
                    "env": manifest_env,
                },
            )
        )

    if skipped:
        logging.warning(
            "Skipping %d MEP member(s) with no checkpoint: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if not refs:
        raise ValueError(f"No loadable MEP population members found in manifest: {manifest_path}")

    refs.sort(key=lambda ref: ref.metadata["member_id"])
    return refs


def population_source_run(population_path: str | Path) -> Path:
    """The MEP run directory that owns this population (the manifest's grandparent).

    ``<run_root>/mep_population/population_manifest.json`` -> ``<run_root>``. Used to
    flag the one run whose numbers against this population are in-distribution.
    """
    return resolve_manifest_path(population_path).parent.parent
