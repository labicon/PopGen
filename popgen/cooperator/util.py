"""Shared loading utilities for cooperator training and evaluation."""

from __future__ import annotations

import json
import logging
import pickle
import re
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from popgen.agents import PolicyCodeAgent
from popgen.agents.reward_policy import RewardPolicyTensorAdapter
from popgen.cooperator.types import PolicyArtifactRef
from popgen.population.pipeline import load_policy_and_env
from popgen.utils.benchmarl_reload import register_dynamic_stubs, reload_experiment
from popgen.utils.imports import import_all_functions_from_file

_ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")
_EGO_POLICY_LOAD_LOCK = threading.Lock()


def _make_reload_save_folder(experiment_folder: Path) -> str:
    """Return an isolated scratch directory for BenchMARL experiment reloads."""
    return tempfile.mkdtemp(prefix=f"reload_{experiment_folder.name}_")


def _reload_experiment_with_callback_fallback(Experiment, checkpoint: Path, experiment_patch: dict[str, Any]):
    """Reload a BenchMARL experiment, tolerating callbacks that cannot be unpickled."""
    return reload_experiment(Experiment, checkpoint, experiment_patch)


# ---------------------------------------------------------------------------
# Artifact selection helpers
# ---------------------------------------------------------------------------


def _read_adherence(evaluation_path: Path) -> float:
    if not evaluation_path.exists():
        return float("-inf")
    try:
        with open(evaluation_path) as f:
            payload = json.load(f)
        return float(payload["overall_metrics"]["adherence"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return float("-inf")


def _infer_artifact_ref(attempt_dir: Path, behavior_idx: int, attempt_idx: int, adherence: float) -> PolicyArtifactRef | None:
    policy_path = attempt_dir / "policy.py"
    if policy_path.exists():
        return PolicyArtifactRef(
            kind="PolicyCode",
            path=policy_path,
            metadata={"behavior_idx": behavior_idx, "attempt_idx": attempt_idx, "adherence": adherence},
        )

    checkpoints = list(attempt_dir.rglob("checkpoints/checkpoint_*.pt"))
    if checkpoints:
        return PolicyArtifactRef(
            kind="RewardFn",
            path=attempt_dir,
            metadata={"behavior_idx": behavior_idx, "attempt_idx": attempt_idx, "adherence": adherence},
        )
    return None


def _latest_summary_path(results_root: Path) -> Path:
    metadata_root = results_root / "metadata"
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Expected current population layout at {results_root}: missing 'metadata/' directory.")

    metadata_runs = sorted([d for d in metadata_root.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not metadata_runs:
        raise FileNotFoundError(f"Expected current population layout at {results_root}: no metadata run directories found.")

    summary_path = metadata_runs[-1] / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Latest metadata summary not found: {summary_path}")

    return summary_path


def _resolve_run_output_dir(results_root: Path, run: dict[str, Any], behavior_idx: int) -> Path | None:
    output_dir_value = run.get("output_dir")
    if isinstance(output_dir_value, str) and output_dir_value:
        output_dir = Path(output_dir_value)
        if not output_dir.is_absolute():
            output_dir = (results_root / output_dir).resolve()
        if output_dir.is_dir():
            return output_dir

    run_timestamp = run.get("run_timestamp")
    if isinstance(run_timestamp, str) and run_timestamp:
        fallback = results_root / "agents" / f"behavior_{behavior_idx}" / run_timestamp
        if fallback.is_dir():
            return fallback

    return None


def select_policy_refs_from_behavior_results(results_dir: str | Path) -> list[PolicyArtifactRef]:
    """Select best attempt per behavior from current population output layout."""
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    agents_root = root / "agents"
    if not agents_root.is_dir():
        raise FileNotFoundError(f"Expected current population layout at {root}: missing 'agents/' directory.")

    summary_path = _latest_summary_path(root)
    try:
        with open(summary_path) as f:
            summary_payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read summary payload at {summary_path}: {exc}") from exc

    runs = summary_payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Invalid summary payload at {summary_path}: expected list field 'runs'.")

    chosen: dict[int, tuple[float, int, PolicyArtifactRef]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue

        behavior_idx_raw = run.get("behavior_idx")
        try:
            behavior_idx = int(behavior_idx_raw)
        except (TypeError, ValueError):
            continue

        run_output_dir = _resolve_run_output_dir(root, run, behavior_idx)
        if run_output_dir is None:
            continue

        candidates: list[tuple[float, int, PolicyArtifactRef]] = []
        skipped_unevaluated: list[str] = []
        for attempt_dir in sorted(run_output_dir.glob("attempt_*")):
            attempt_match = _ATTEMPT_RE.match(attempt_dir.name)
            if attempt_match is None or not attempt_dir.is_dir():
                continue
            attempt_idx = int(attempt_match.group(1))

            # An attempt with no evaluation.json never produced a judged rollout:
            # training diverged, or generation failed every retry. It can still
            # leave artifacts behind -- BenchMARL checkpoints written before the
            # crash, or a policy.py from the last failed generation -- and those
            # used to be selectable with adherence=-inf. When every attempt for a
            # behavior failed that way, the cooperator silently trained against a
            # checkpoint from a diverged run.
            if not (attempt_dir / "evaluation.json").exists():
                skipped_unevaluated.append(attempt_dir.name)
                continue

            adherence = _read_adherence(attempt_dir / "evaluation.json")
            ref = _infer_artifact_ref(attempt_dir, behavior_idx, attempt_idx, adherence)
            if ref is None:
                continue
            candidates.append((adherence, attempt_idx, ref))

        if skipped_unevaluated:
            logging.warning(
                "behavior_%s: skipped %d unevaluated attempt(s) with no evaluation.json (%s).",
                behavior_idx,
                len(skipped_unevaluated),
                ", ".join(skipped_unevaluated),
            )

        if not candidates:
            logging.warning(
                "behavior_%s: no evaluated attempt found in %s; this behavior is excluded from the population.",
                behavior_idx,
                run_output_dir,
            )
            continue

        best_candidate = max(candidates, key=lambda item: (item[0], item[1]))
        previous = chosen.get(behavior_idx)
        if previous is None or (best_candidate[0], best_candidate[1]) > (previous[0], previous[1]):
            chosen[behavior_idx] = best_candidate

    refs: list[PolicyArtifactRef] = []
    for behavior_idx in sorted(chosen.keys()):
        _, _, ref = chosen[behavior_idx]
        refs.append(ref)

    if not refs:
        raise ValueError(
            f"No loadable teammate policy artifacts found in current layout under {root}. "
            "Expected metadata runs with attempt directories containing policy.py or checkpoints, "
            "and an evaluation.json recording the behavior-adherence score. Attempts whose training "
            "diverged or whose generation failed every retry have no evaluation.json and are "
            "deliberately excluded; see the per-behavior warnings above."
        )

    return refs


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def load_policies(refs: list[PolicyArtifactRef], env_module: Any, device: str | None = None) -> list[nn.Module]:
    """Load teammate policies from a list of artifact references.

    ``device`` places RewardFn teammates; it defaults to CUDA when available
    because cooperator training runs them once per step over many parallel
    envs. PolicyCode teammates are plain callables and ignore it.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policies: list[nn.Module] = []
    for ref in refs:
        if ref.kind == "PolicyCode":
            fns = import_all_functions_from_file(str(ref.path))
            if "policy" not in fns:
                raise KeyError(f"No 'policy' function found in {ref.path}")
            policies.append(PolicyCodeAgent(fns["policy"]))
        elif ref.kind == "RewardFn":
            # load_policy_and_env always returns a (env, policy) tuple.
            env, ego_policy = load_policy_and_env(
                policy_type="RewardFn",
                create_env_fn=env_module.create_env,
                save_dir=str(ref.path),
                reload_context=getattr(env_module, "get_cooperator_reload_context", None),
                device=device,
            )
            # Teammate routing calls policies with a plain (batch, obs_dim)
            # tensor; a reloaded BenchMARL actor needs a nested TensorDict.
            policies.append(RewardPolicyTensorAdapter(ego_policy))
            logging.info("Loaded RewardFn teammate for behavior_%s on %s.", ref.metadata.get("behavior_idx"), device)
            if env is not None and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
        else:
            raise ValueError(f"Unsupported artifact kind: {ref.kind}")

    if not policies:
        raise ValueError("No teammate policies were loaded from selected artifacts.")
    return policies


# ---------------------------------------------------------------------------
# Stub-aware ego policy loader (needed for standalone re-evaluation)
# ---------------------------------------------------------------------------


def load_ego_policy_and_env(save_dir: str, create_env_fn: Callable) -> tuple[Any, Any]:
    """Load the ego policy and test env from a BenchMARL checkpoint directory.

    Uses a custom unpickler so pickled callbacks referencing popgen._dynamic
    modules don't raise ModuleNotFoundError when reloading a trained checkpoint
    in a fresh Python process.

    Returns:
        ``(ego_policy, env)``
    """
    from benchmarl.experiment import Experiment

    with _EGO_POLICY_LOAD_LOCK:
        StubUnpickler = register_dynamic_stubs()

        _orig_pickle_load = pickle.load

        def _patched_load(f, **kwargs):
            return StubUnpickler(f).load()

        pickle.load = _patched_load
        try:
            save_dir_path = Path(save_dir)
            checkpoints = sorted(
                list(save_dir_path.rglob("checkpoints/checkpoint_*.pt")),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {save_dir_path}")

            checkpoint = checkpoints[-1]
            experiment_folder = checkpoint.parent.parent.resolve()
            experiment_patch = {
                # BenchMARL rewrites config.pkl during reload; keep that away from the source artifact.
                "save_folder": _make_reload_save_folder(experiment_folder),
                "loggers": [],
                "create_json": False,
                "evaluation_episodes": 1,  # enforce single-env rollout to avoid multi-episode trajectories
            }
            if not torch.cuda.is_available():
                experiment_patch["sampling_device"] = "cpu"
                experiment_patch["train_device"] = "cpu"
                experiment_patch["restore_map_location"] = {"cuda:0": "cpu"}
                if str(experiment_patch.get("buffer_device", "")).lower() != "disk":
                    experiment_patch["buffer_device"] = "cpu"

            experiment = _reload_experiment_with_callback_fallback(
                Experiment,
                checkpoint,
                experiment_patch,
            )
        finally:
            pickle.load = _orig_pickle_load

    ego_group = "ego" if "ego" in experiment.group_policies else "agents"
    env = experiment.test_env
    env_device = getattr(env, "device", "cpu")
    ego_policy = experiment.group_policies[ego_group].to(env_device)
    # Tag with training group so compose_rollout_policy can pick the right wrapper.
    ego_policy._benchmarl_ego_group = ego_group
    return ego_policy, env
