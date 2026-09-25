"""Cooperator evaluation — standalone, no re-training required."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import logging
import re
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any

import torch
import torch.nn as nn
from pqdm.processes import pqdm
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from popgen.agents.router import MultiPolicyRouter
from popgen.cooperator.util import load_ego_policy_and_env, load_policies, select_policy_refs_from_behavior_results
from popgen.utils.logging import save_rollout
from popgen.utils.run_names import infer_env_name

# ---------------------------------------------------------------------------
# Rollout result helpers
# ---------------------------------------------------------------------------


def _extract_ego_team_reward_breakdown(states: list[dict[str, Any]]) -> dict[str, float | None]:
    team_total = 0.0
    ego_total = 0.0
    teammate_total = 0.0
    found_team = False
    found_ego = False
    found_teammate = False

    for step in states:
        for agent_name, agent in step.get("ego_team", {}).items():
            r = agent.get("reward")
            if r is not None:
                reward = float(r)
                team_total += reward
                found_team = True
                if agent_name == "ego_agent":
                    ego_total += reward
                    found_ego = True
                else:
                    teammate_total += reward
                    found_teammate = True
    return {
        "team_reward": team_total if found_team else None,
        "ego_reward": ego_total if found_ego else None,
        "teammate_reward": teammate_total if found_teammate else None,
    }


def _build_scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": variance**0.5, "min": min(values), "max": max(values)}


def _stringify_rollout_error(error: Any) -> str:
    if isinstance(error, BaseException):
        detail = str(error).strip() or "<no message>"
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
        return f"{type(error).__name__}: {detail}\n{tb}" if tb else f"{type(error).__name__}: {detail}"
    return str(error)


def _build_rollout_error_result(rollout_idx: int, error: Any) -> dict[str, Any]:
    return {
        "rollout_idx": rollout_idx,
        "rollout_error": None if error is None else _stringify_rollout_error(error),
        "trajectory_length": 0,
        "rollout_path": None,
        "team_reward": None,
        "ego_reward": None,
        "teammate_reward": None,
        "elapsed_sec": 0.0,
    }


def _build_rollout_summary(
    n_eval_rollouts: int,
    rollout_errors: list[str | None],
    trajectory_lengths: list[int],
    rollout_paths: list[str | None],
    per_rollout_team_rewards: list[float | None],
    per_rollout_ego_rewards: list[float | None],
    per_rollout_teammate_rewards: list[float | None],
    elapsed: float,
) -> dict:
    n_failed = sum(1 for e in rollout_errors if e is not None)
    avg_traj_len = (sum(trajectory_lengths) / len(trajectory_lengths)) if trajectory_lengths else 0.0
    team_reward_values = [r for r in per_rollout_team_rewards if r is not None]
    ego_reward_values = [r for r in per_rollout_ego_rewards if r is not None]
    teammate_reward_values = [r for r in per_rollout_teammate_rewards if r is not None]
    return {
        "n_rollouts": n_eval_rollouts,
        "n_successful_rollouts": n_eval_rollouts - n_failed,
        "n_failed_rollouts": n_failed,
        "failure_rate": n_failed / n_eval_rollouts,
        "avg_trajectory_length": avg_traj_len,
        "reward_summary": _build_scalar_summary(team_reward_values),
        "ego_reward_summary": _build_scalar_summary(ego_reward_values),
        "teammate_reward_summary": _build_scalar_summary(teammate_reward_values),
        "per_rollout_reward": per_rollout_team_rewards,
        "per_rollout_ego_reward": per_rollout_ego_rewards,
        "per_rollout_teammate_reward": per_rollout_teammate_rewards,
        "rollout_errors": rollout_errors,
    }


# ---------------------------------------------------------------------------
# Null-action ego policy (used by BaselineEvaluator)
# ---------------------------------------------------------------------------


class NullActionEgoPolicy(TensorDictModuleBase):
    """Ego policy that always outputs null (zero) actions.

    Used by :class:`BaselineEvaluator` to measure teammate performance without
    any cooperator contribution.
    """

    in_keys = [("ego", "observation")]
    out_keys = [("ego", "action")]

    def __init__(self, ego_env: Any):
        super().__init__()
        self._benchmarl_ego_group = "ego"
        try:
            spec = ego_env.full_action_spec["ego"]["action"]
            n_batch_dims = len(ego_env.batch_size)
            self._action_shape = tuple(spec.shape[n_batch_dims + 1 :])
            self._action_dtype = spec.dtype
        except (KeyError, AttributeError):
            self._action_shape = ()
            self._action_dtype = torch.long

    def forward(self, input_td: TensorDictBase) -> TensorDictBase:
        ego_obs = input_td["ego"]["observation"]
        prefix_shape = ego_obs.shape[:-1]
        null_action = torch.zeros(
            prefix_shape + self._action_shape,
            dtype=self._action_dtype,
            device=ego_obs.device,
        )
        input_td.set(("ego", "action"), null_action)
        return input_td


# ---------------------------------------------------------------------------
# Teammate exploration
# ---------------------------------------------------------------------------

#: How a teammate policy picks actions during evaluation rollouts.
#: ``"random"`` samples from the policy distribution, which is how every neural
#: teammate behaved while the ego was collecting experience against it;
#: ``"deterministic"`` takes the distribution mode; ``None`` inherits whatever
#: the surrounding rollout context happens to be.
_TEAMMATE_EXPLORATION_TYPES: dict[str, ExplorationType] = {
    "random": ExplorationType.RANDOM,
    "deterministic": ExplorationType.DETERMINISTIC,
}
DEFAULT_TEAMMATE_EXPLORATION = "random"


def resolve_teammate_exploration(mode: str | None) -> ExplorationType | None:
    """Map a config string onto an :class:`ExplorationType`, or ``None`` to inherit."""
    if mode is None or (isinstance(mode, str) and mode.strip().lower() in {"", "inherit", "none"}):
        return None
    key = str(mode).strip().lower()
    if key not in _TEAMMATE_EXPLORATION_TYPES:
        raise ValueError(f"teammate_exploration must be one of {sorted(_TEAMMATE_EXPLORATION_TYPES)} or 'inherit', got {mode!r}")
    return _TEAMMATE_EXPLORATION_TYPES[key]


class PinnedExplorationPolicy(nn.Module):
    """Run a teammate policy under its own exploration type.

    ``set_exploration_type`` is a process-global context, so the
    ``DETERMINISTIC`` block that pins the ego to its greedy action also pins
    every neural teammate to its distribution mode. That is not the partner
    the ego trained against: teammates are collected against under
    ``ExplorationType.RANDOM`` (torchrl's collector default, and BenchMARL's
    ``experiment.py`` collection loop), and undertrained policies whose action
    distributions are still flat degenerate badly under argmax -- in LBF they
    deadlock and time out. Forcing argmax therefore penalises exactly the
    methods whose partners are neural (self-play, FCP, MEP) while leaving
    scripted partners untouched, which is not a like-for-like comparison.

    Scripted and policy-code teammates ignore the exploration type entirely,
    so wrapping them is a no-op.
    """

    def __init__(self, policy: nn.Module, exploration_type: ExplorationType):
        super().__init__()
        self.policy = policy
        self.exploration_type = exploration_type

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with set_exploration_type(self.exploration_type):
            return self.policy(*args, **kwargs)

    def __repr__(self) -> str:
        return f"PinnedExplorationPolicy({self.policy!r}, exploration_type={self.exploration_type})"


def pin_teammate_exploration(teammate: nn.Module, mode: str | None) -> nn.Module:
    """Wrap *teammate* so it keeps sampling regardless of the rollout context."""
    exploration_type = resolve_teammate_exploration(mode)
    if exploration_type is None:
        return teammate
    return PinnedExplorationPolicy(teammate, exploration_type)


# ---------------------------------------------------------------------------
# Teammate loading
# ---------------------------------------------------------------------------


def _load_teammate(
    teammate_spec: dict[str, Any],
    create_env_fn: Callable[..., Any],
    env_module: Any,
) -> Any:
    """Load a teammate policy from a spec dict.

    Supported kinds:
    - ``"scripted_factory"``: instantiate scripted policy from env.
    - ``"policy_ref"``: load a :class:`PolicyArtifactRef` (policy-code / reward-fn).
    - ``"selfplay_checkpoint"``: load a self-play seed directory checkpoint.
    """
    kind = teammate_spec["kind"]

    if kind == "scripted_factory":
        temp_env = create_env_fn(num_envs=1, seed=0, device="cpu")
        try:
            return teammate_spec["factory"](temp_env)
        finally:
            if hasattr(temp_env, "close"):
                try:
                    temp_env.close()
                except Exception:
                    pass

    if kind == "policy_ref":
        return load_policies([teammate_spec["ref"]], env_module)[0]

    if kind == "selfplay_checkpoint":
        from popgen.cooperator.selfplay_util import SelfPlayPolicyTensorAdapter, load_selfplay_policy

        reload_context = getattr(env_module, "get_cooperator_reload_context", None)
        wrapper = load_selfplay_policy(teammate_spec["seed_dir"], reload_context=reload_context)
        return SelfPlayPolicyTensorAdapter(wrapper)

    raise ValueError(f"Unsupported teammate spec kind: {kind!r}")


# ---------------------------------------------------------------------------
# Rollout workers
# ---------------------------------------------------------------------------


def _run_cooperator_eval_worker(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Load ego and teammate policies once, then run all rollouts sequentially.

    Parallelism is across teammates (one worker per teammate), not across
    rollouts. This avoids reloading checkpoints on every rollout call.

    Each rollout receives its own seed via ``args["rollout_seeds"][i]``.
    ``rollout_fn`` calls ``apply_rollout_seed(env, seed)`` at entry, which
    resets both the env RNG and ``torch.manual_seed``, so sequential reuse
    of the same env object is deterministic and independent across rollouts.

    Returns a list of per-rollout result dicts.
    """
    ego_env = None
    try:
        env_module = importlib.import_module(args["env_module_name"])
        create_env_fn = env_module.create_env
        rollout_fn = env_module.rollout_fn
        serialize_state_fn = getattr(env_module, "serialize_state_fn", None)

        get_reload_ctx = getattr(env_module, "get_cooperator_reload_context", None)
        reload_ctx = get_reload_ctx() if callable(get_reload_ctx) else nullcontext()
        with reload_ctx:
            ego_policy, ego_env = load_ego_policy_and_env(args["cooperator_dir"], create_env_fn)

        teammate = pin_teammate_exploration(
            _load_teammate(args["teammate_spec"], create_env_fn, env_module),
            args.get("teammate_exploration", DEFAULT_TEAMMATE_EXPLORATION),
        )
        rollout_policy = MultiPolicyRouter({"ego": ego_policy, "teammate": [teammate]})

        results: list[dict[str, Any]] = []
        for rollout_idx, (seed, rollout_path) in enumerate(zip(args["rollout_seeds"], args["rollout_paths"])):
            step_start = time()
            states = images = None
            try:
                with set_exploration_type(ExplorationType.DETERMINISTIC):
                    states, images, error = rollout_fn(rollout_policy, env=ego_env, seed=seed)
            except Exception as exc:
                states, images, error = None, None, exc

            reward_breakdown = {
                "team_reward": None,
                "ego_reward": None,
                "teammate_reward": None,
            }
            if error is None and states:
                reward_breakdown = _extract_ego_team_reward_breakdown(states)

            rollout_error = None if error is None else _stringify_rollout_error(error)
            if rollout_error is None and states is not None:
                rollout_data = deepcopy(states)
                if callable(serialize_state_fn):
                    rollout_data = serialize_state_fn(rollout_data)
                save_rollout(rollout_data=rollout_data, images=images or [], save_path=rollout_path)

            results.append(
                {
                    "rollout_idx": rollout_idx,
                    "rollout_error": rollout_error,
                    "trajectory_length": len(states) if rollout_error is None and states is not None else 0,
                    "rollout_path": rollout_path if rollout_error is None and states is not None else None,
                    "team_reward": reward_breakdown["team_reward"],
                    "ego_reward": reward_breakdown["ego_reward"],
                    "teammate_reward": reward_breakdown["teammate_reward"],
                    "elapsed_sec": time() - step_start,
                }
            )
            states = images = None  # release references between rollouts

        return results
    except Exception as exc:
        return [_build_rollout_error_result(i, exc) for i in range(len(args["rollout_seeds"]))]
    finally:
        if ego_env is not None and hasattr(ego_env, "close"):
            try:
                ego_env.close()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_baseline_rollout_worker(args: dict[str, Any]) -> dict[str, Any]:
    started_at = time()
    baseline_env = None
    states = None
    images = None
    try:
        env_module = importlib.import_module(args["env_module_name"])
        create_env_fn = env_module.create_env
        rollout_fn = env_module.rollout_fn
        serialize_state_fn = getattr(env_module, "serialize_state_fn", None)

        baseline_env = create_env_fn(num_envs=1, seed=0, device="cpu")
        null_ego_policy = NullActionEgoPolicy(baseline_env)

        teammate = pin_teammate_exploration(
            _load_teammate(args["teammate_spec"], create_env_fn, env_module),
            args.get("teammate_exploration", DEFAULT_TEAMMATE_EXPLORATION),
        )
        rollout_policy = MultiPolicyRouter({"ego": null_ego_policy, "teammate": [teammate]})

        try:
            states, images, error = rollout_fn(rollout_policy, env=baseline_env, **dict(args["eval_kwargs"]))
        except Exception as exc:
            states, images, error = None, None, exc

        reward_breakdown = {
            "team_reward": None,
            "ego_reward": None,
            "teammate_reward": None,
        }
        if error is None and states:
            reward_breakdown = _extract_ego_team_reward_breakdown(states)

        rollout_error = None if error is None else _stringify_rollout_error(error)
        if rollout_error is None and states is not None:
            rollout_data = deepcopy(states)
            if callable(serialize_state_fn):
                rollout_data = serialize_state_fn(rollout_data)
            save_rollout(rollout_data=rollout_data, images=images or [], save_path=args["rollout_path"])

        return {
            "rollout_idx": args["rollout_idx"],
            "rollout_error": rollout_error,
            "trajectory_length": len(states) if rollout_error is None and states is not None else 0,
            "rollout_path": args["rollout_path"] if rollout_error is None and states is not None else None,
            "team_reward": reward_breakdown["team_reward"],
            "ego_reward": reward_breakdown["ego_reward"],
            "teammate_reward": reward_breakdown["teammate_reward"],
            "elapsed_sec": time() - started_at,
        }
    except Exception as exc:
        return {
            "rollout_idx": args["rollout_idx"],
            "rollout_error": _stringify_rollout_error(exc),
            "trajectory_length": 0,
            "rollout_path": None,
            "team_reward": None,
            "ego_reward": None,
            "teammate_reward": None,
            "elapsed_sec": time() - started_at,
        }
    finally:
        if baseline_env is not None and hasattr(baseline_env, "close"):
            try:
                baseline_env.close()
            except Exception:
                pass
        baseline_env = None
        states = None
        images = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_rollouts(
    worker_fn: Callable,
    n_eval_rollouts: int,
    env_module: Any,
    eval_dir: Path,
    eval_kwargs: dict,
    get_teammate_spec: Callable[[int], dict[str, Any]],
    rollout_dir_fn: Callable[[int], str],
    num_workers: int,
    extra_worker_args: dict | None = None,
) -> tuple[
    list[str | None],
    list[int],
    list[str | None],
    list[float | None],
    list[float | None],
    list[float | None],
    float,
]:
    """Run rollouts in parallel via pqdm and return collected metrics.

    Returns:
        ``(rollout_errors, trajectory_lengths, rollout_paths,
        per_rollout_team_rewards, per_rollout_ego_rewards,
        per_rollout_teammate_rewards, elapsed_sec)``
    """
    started_at = time()
    base_seed = int(eval_kwargs.get("seed", 0))
    env_module_name = getattr(env_module, "__name__", None)
    if env_module_name is None:
        raise TypeError("_run_rollouts requires an importable env module (needs __name__).")

    rollout_args = []
    for rollout_idx in range(n_eval_rollouts):
        call_kwargs = dict(eval_kwargs)
        call_kwargs["seed"] = base_seed + rollout_idx * 1000
        entry: dict[str, Any] = {
            "rollout_idx": rollout_idx,
            "env_module_name": env_module_name,
            "teammate_spec": get_teammate_spec(rollout_idx),
            "rollout_path": str(eval_dir / rollout_dir_fn(rollout_idx)),
            "eval_kwargs": call_kwargs,
        }
        if extra_worker_args:
            entry.update(extra_worker_args)
        rollout_args.append(entry)

    raw_results = pqdm(rollout_args, worker_fn, n_jobs=num_workers, desc="Running eval rollouts")

    normalized: list[dict[str, Any]] = []
    for rollout_idx, result in enumerate(raw_results):
        if isinstance(result, BaseException):
            normalized.append(_build_rollout_error_result(rollout_idx, result))
        elif isinstance(result, dict):
            normalized.append(result)
        else:
            normalized.append(
                _build_rollout_error_result(
                    rollout_idx,
                    TypeError(f"Unexpected pqdm result type: {type(result).__name__}"),
                )
            )

    rollout_errors = [r["rollout_error"] for r in normalized]
    trajectory_lengths = [r["trajectory_length"] for r in normalized if r["rollout_error"] is None]
    rollout_paths = [r["rollout_path"] for r in normalized]
    per_rollout_team_rewards = [r["team_reward"] for r in normalized]
    per_rollout_ego_rewards = [r["ego_reward"] for r in normalized]
    per_rollout_teammate_rewards = [r["teammate_reward"] for r in normalized]

    return (
        rollout_errors,
        trajectory_lengths,
        rollout_paths,
        per_rollout_team_rewards,
        per_rollout_ego_rewards,
        per_rollout_teammate_rewards,
        time() - started_at,
    )


def run_baseline_rollouts(
    env_module: Any,
    n_eval_rollouts: int,
    eval_dir: Path,
    eval_kwargs: dict,
    get_teammate_spec: Callable[[int], dict[str, Any]],
    rollout_dir_fn: Callable[[int], str],
    num_workers: int,
) -> tuple[
    list[str | None],
    list[int],
    list[str | None],
    list[float | None],
    list[float | None],
    list[float | None],
    float,
]:
    return _run_rollouts(
        worker_fn=_run_baseline_rollout_worker,
        n_eval_rollouts=n_eval_rollouts,
        env_module=env_module,
        eval_dir=eval_dir,
        eval_kwargs=eval_kwargs,
        get_teammate_spec=get_teammate_spec,
        rollout_dir_fn=rollout_dir_fn,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Scripted policy resolution
# ---------------------------------------------------------------------------


def _resolve_scripted_policies(env_module: Any, evaluation_set: str) -> list[Callable]:
    normalized = evaluation_set.strip().lower().replace("-", "_")

    if hasattr(env_module, "get_named_evaluation_scripted_policies"):
        policies = env_module.get_named_evaluation_scripted_policies(normalized)
        if isinstance(policies, list) and policies:
            return policies
        raise ValueError("get_named_evaluation_scripted_policies() must return a non-empty list.")

    if normalized in {"standard", "default"} and hasattr(env_module, "get_evaluation_scripted_policies"):
        policies = env_module.get_evaluation_scripted_policies()
        if isinstance(policies, list) and policies:
            return policies
        raise ValueError("get_evaluation_scripted_policies() must return a non-empty list.")

    # Legacy fallbacks
    attr = "EVALUATION_BEHAVIOR_SET" if normalized in {"standard", "default"} else f"{normalized.upper()}_EVALUATION_BEHAVIOR_SET"
    if hasattr(env_module, attr):
        behavior_set = getattr(env_module, attr)
        if isinstance(behavior_set, list):
            policies = [e.get("policy_cls") for e in behavior_set if isinstance(e, dict)]
            policies = [p for p in policies if p is not None]
            if policies:
                return policies

    if normalized in {"standard", "default"} and hasattr(env_module, "ScriptedPolicy"):
        return [env_module.ScriptedPolicy]

    raise AttributeError(f"env_module does not expose scripted policies for evaluation_set={evaluation_set!r}.")


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _build_teammate_eval_summary(
    n_eval_rollouts: int,
    teammate_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    all_team_rewards = [r for entry in teammate_entries for r in entry["per_rollout_reward"] if r is not None]
    all_ego_rewards = [r for entry in teammate_entries for r in entry["per_rollout_ego_reward"] if r is not None]
    all_teammate_rewards = [r for entry in teammate_entries for r in entry["per_rollout_teammate_reward"] if r is not None]
    return {
        "n_teammates": len(teammate_entries),
        "n_eval_rollouts": n_eval_rollouts,
        "aggregate_reward_summary": _build_scalar_summary(all_team_rewards),
        "aggregate_ego_reward_summary": _build_scalar_summary(all_ego_rewards),
        "aggregate_teammate_reward_summary": _build_scalar_summary(all_teammate_rewards),
        "teammates": teammate_entries,
        # A teammate whose policy failed to load returns only error rollouts, so it
        # contributes nothing to the aggregates above while still counting in
        # n_teammates -- downstream that becomes NaN and is meaned away, making a
        # half-broken run indistinguishable from a clean one. Count them explicitly.
        "n_failed_teammates": sum(1 for entry in teammate_entries if entry.get("n_successful_rollouts", 0) == 0),
        "checkpoint_source": "latest",
    }


def _save_summary(output_dir: Path, filename: str, summary: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(path)
    return summary


def _sanitize_path_fragment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return normalized or "unnamed"


def _build_population_source_tag(population_dir: Path) -> str:
    readable_parts = [_sanitize_path_fragment(part) for part in population_dir.parts[-2:]]
    readable = "__".join(part for part in readable_parts if part)
    digest = hashlib.sha1(str(population_dir).encode("utf-8")).hexdigest()[:8]
    return f"{readable}__{digest}" if readable else digest


def _build_rollout_subdir_name(
    *,
    output_filename: str,
    eval_type: str,
    population_dir: Path | None,
) -> str:
    output_path = Path(output_filename)
    fragments = [_sanitize_path_fragment(output_path.stem)]
    if eval_type in _POPULATION_EVAL_TYPES and population_dir is not None:
        fragments.append(_build_population_source_tag(population_dir))
    fragments.append("rollouts")
    return "__".join(fragments)


# ---------------------------------------------------------------------------
# CooperatorEvaluator
# ---------------------------------------------------------------------------

# Fixed output filenames only for unambiguous heldout sets.
# policy_code and self_play have no fixed default: the caller must supply
# output_filename to evaluate() so the semantic relationship (training vs.
# cross-evaluation) is made explicit at the call site.
_EVAL_TYPE_DEFAULT_OUTPUT_FILE: dict[str, str] = {
    "eval_teammate": "eval_teammates.json",
}
_EVAL_TYPE_ROLLOUT_PREFIX: dict[str, str] = {
    "eval_teammate": "eval_teammate",
    "policy_code": "teammate",
    "self_play": "selfplay_teammate",
    "mep_population": "mep_member",
}
# Eval types whose teammates come from a population on disk. These require
# population_dir, and tag their rollout directory with the population's identity so
# two populations evaluated against the same cooperator cannot collide.
_POPULATION_EVAL_TYPES: frozenset[str] = frozenset({"policy_code", "self_play", "mep_population"})
VALID_EVAL_TYPES: frozenset[str] = frozenset({"eval_teammate"} | _POPULATION_EVAL_TYPES)


class CooperatorEvaluator:
    """Evaluates a trained cooperator checkpoint against a configured teammate set.

    ``eval_type`` selects the teammate source at construction time:

    * ``"eval_teammate"`` — heldout scripted behaviors from env_module (standard set).
    * ``"policy_code"``  — policy-code / reward-fn population; requires ``population_dir``.
    * ``"self_play"``    — self-play checkpoint population; requires ``population_dir``.
                           Handles the ``"agents"`` group mismatch automatically via
                           :class:`~popgen.cooperator.selfplay_util.SelfPlayPolicyTensorAdapter`.

    ``teammate_exploration`` controls how teammate policies pick actions,
    independently of the ``DETERMINISTIC`` context that pins the ego. It
    defaults to ``"random"`` (sample), matching how neural teammates behaved
    while the ego was training against them -- see
    :class:`PinnedExplorationPolicy` for why this must not follow the ego.

    Call :meth:`evaluate` to run rollouts and write results under
    ``<cooperator_dir>/evaluation/<output_file>``.
    """

    def __init__(
        self,
        *,
        cooperator_dir: str | Path,
        env_module: Any,
        eval_type: str,
        population_dir: str | Path | None = None,
        num_workers: int = 8,
        teammate_exploration: str | None = DEFAULT_TEAMMATE_EXPLORATION,
    ):
        if eval_type not in VALID_EVAL_TYPES:
            raise ValueError(f"eval_type must be one of {sorted(VALID_EVAL_TYPES)}, got {eval_type!r}")
        if eval_type in _POPULATION_EVAL_TYPES and population_dir is None:
            raise ValueError(f"eval_type={eval_type!r} requires population_dir to be set.")
        self.cooperator_dir = Path(cooperator_dir).resolve()
        self.env_module = env_module
        self.eval_type = eval_type
        self.population_dir = Path(population_dir).resolve() if population_dir is not None else None
        self.num_workers = max(1, int(num_workers))
        # Validate eagerly so a bad config fails here rather than inside every worker.
        resolve_teammate_exploration(teammate_exploration)
        self.teammate_exploration = teammate_exploration

    @property
    def _eval_dir(self) -> Path:
        return self.cooperator_dir / "evaluation"

    def _rollout_root_dir(self, output_filename: str) -> Path:
        output_path = Path(output_filename)
        return (
            self._eval_dir
            / output_path.parent
            / _build_rollout_subdir_name(
                output_filename=output_filename,
                eval_type=self.eval_type,
                population_dir=self.population_dir,
            )
        )

    def _load_teammate_specs(self) -> list[dict[str, Any]]:
        """Return a list of picklable teammate spec dicts for the configured eval_type."""
        if self.eval_type == "eval_teammate":
            policies = _resolve_scripted_policies(self.env_module, "standard")
            return [
                {
                    "kind": "scripted_factory",
                    "factory": p,
                    "name": getattr(p, "__name__", f"{self.eval_type}_{i}"),
                }
                for i, p in enumerate(policies)
            ]

        if self.eval_type == "policy_code":
            refs = select_policy_refs_from_behavior_results(self.population_dir)
            return [{"kind": "policy_ref", "ref": ref, "path": str(ref.path)} for ref in refs]

        if self.eval_type == "mep_population":
            # MEP members are ordinary RewardFn artifacts, so they reuse the
            # "policy_ref" loader; only the enumeration is MEP-specific.
            from popgen.cooperator.mep_util import enumerate_mep_member_refs

            refs = enumerate_mep_member_refs(
                self.population_dir,
                expected_env=infer_env_name(self.env_module),
            )
            return [
                {
                    "kind": "policy_ref",
                    "ref": ref,
                    "path": str(ref.path),
                    "name": f"member_{ref.metadata['member_id']}",
                    "member_id": ref.metadata["member_id"],
                }
                for ref in refs
            ]

        # self_play
        from popgen.cooperator.selfplay_util import enumerate_selfplay_seed_dirs

        seed_dirs = enumerate_selfplay_seed_dirs(self.population_dir)
        return [{"kind": "selfplay_checkpoint", "seed_dir": str(sd)} for sd in seed_dirs]

    def evaluate(self, n_eval_rollouts: int = 50, output_filename: str | None = None) -> dict:
        """Evaluate the cooperator against all teammates of the configured type.

        Writes results to ``<cooperator_dir>/evaluation/<output_filename>`` and
        returns the summary dict (augmented with ``summary_path``).

        ``output_filename`` is required for ``eval_type`` in ``{"policy_code", "self_play"}``
        because the semantic relationship (training vs. cross-evaluation) depends on context
        and cannot be inferred from the teammate source alone. For ``"eval_teammate"`` it
        defaults to ``eval_teammates.json``.
        """
        if output_filename is None:
            if self.eval_type not in _EVAL_TYPE_DEFAULT_OUTPUT_FILE:
                raise ValueError(f"output_filename is required for eval_type={self.eval_type!r}. Use e.g. 'training_teammates.json' or 'cross_eval_policy_code.json'.")
            output_filename = _EVAL_TYPE_DEFAULT_OUTPUT_FILE[self.eval_type]

        env_module_name = getattr(self.env_module, "__name__", None)
        if env_module_name is None:
            raise TypeError("CooperatorEvaluator requires an importable env module (needs __name__).")

        teammate_specs = self._load_teammate_specs()
        rollout_prefix = _EVAL_TYPE_ROLLOUT_PREFIX[self.eval_type]
        rollout_seeds = [i * 1000 for i in range(n_eval_rollouts)]
        rollout_root_dir = self._rollout_root_dir(output_filename)

        logging.info(
            "CooperatorEvaluator[%s]: %d teammates, %d rollouts each, teammate_exploration=%s → %s",
            self.eval_type,
            len(teammate_specs),
            n_eval_rollouts,
            self.teammate_exploration,
            output_filename,
        )

        # One pqdm job per teammate: load policies once, run all rollouts sequentially.
        # This avoids the per-rollout checkpoint reload cost of the old design.
        worker_args = [
            {
                "teammate_spec": spec,
                "cooperator_dir": str(self.cooperator_dir),
                "env_module_name": env_module_name,
                "rollout_seeds": rollout_seeds,
                "teammate_exploration": self.teammate_exploration,
                "rollout_paths": [str(rollout_root_dir / f"rollout_{i}_with_{rollout_prefix}_{tm_idx}") for i in range(n_eval_rollouts)],
            }
            for tm_idx, spec in enumerate(teammate_specs)
        ]

        raw_results = pqdm(
            worker_args,
            _run_cooperator_eval_worker,
            n_jobs=self.num_workers,
            desc=f"Evaluating [{self.eval_type}] teammates",
        )

        teammate_entries: list[dict] = []
        for tm_idx, (spec, rollout_results) in enumerate(zip(teammate_specs, raw_results)):
            if isinstance(rollout_results, BaseException):
                rollout_results = [_build_rollout_error_result(i, rollout_results) for i in range(n_eval_rollouts)]
            elif not isinstance(rollout_results, list):
                rollout_results = [_build_rollout_error_result(i, TypeError(f"Unexpected worker result: {type(rollout_results).__name__}")) for i in range(n_eval_rollouts)]

            rollout_errors = [r["rollout_error"] for r in rollout_results]
            traj_lengths = [r["trajectory_length"] for r in rollout_results if r["rollout_error"] is None]
            rollout_paths = [r["rollout_path"] for r in rollout_results]
            per_rollout_team_rewards = [r["team_reward"] for r in rollout_results]
            per_rollout_ego_rewards = [r["ego_reward"] for r in rollout_results]
            per_rollout_teammate_rewards = [r["teammate_reward"] for r in rollout_results]
            elapsed = sum(r.get("elapsed_sec", 0.0) for r in rollout_results)

            entry = _build_rollout_summary(
                n_eval_rollouts,
                rollout_errors,
                traj_lengths,
                rollout_paths,
                per_rollout_team_rewards,
                per_rollout_ego_rewards,
                per_rollout_teammate_rewards,
                elapsed,
            )
            entry["idx"] = tm_idx
            if "name" in spec:
                entry["name"] = spec["name"]
            if spec.get("kind") == "selfplay_checkpoint":
                entry["seed_dir"] = spec["seed_dir"]
            if spec.get("kind") == "policy_ref":
                entry["policy_path"] = spec["path"]
            if "member_id" in spec:
                entry["member_id"] = spec["member_id"]
            teammate_entries.append(entry)

        summary = _build_teammate_eval_summary(n_eval_rollouts, teammate_entries)
        # Stamped so a summary written before the teammate-exploration fix is
        # distinguishable on disk from one written after it.
        summary["teammate_exploration"] = self.teammate_exploration
        summary["ego_exploration"] = "deterministic"
        summary["evaluation_kind"] = self.eval_type
        if self.population_dir is not None:
            summary["population_dir"] = str(self.population_dir)
        if self.eval_type == "mep_population":
            from popgen.cooperator.mep_util import population_source_run, resolve_manifest_path

            summary["population_manifest_path"] = str(resolve_manifest_path(self.population_dir))
            # The MEP run that owns this population: its own numbers against it are
            # in-distribution, so plots must be able to mark that bar.
            summary["population_source_run"] = str(population_source_run(self.population_dir))
        return _save_summary(self._eval_dir, output_filename, summary)


# ---------------------------------------------------------------------------
# BaselineEvaluator
# ---------------------------------------------------------------------------


class BaselineEvaluator:
    """Evaluates reward when the ego agent performs null (zero) actions.

    Provides a baseline to compare against :class:`CooperatorEvaluator` —
    the difference in team reward between the two isolates the
    cooperator's actual contribution to team performance.
    """

    def __init__(self, *, output_dir: str | Path, env_module: Any, num_workers: int = 8):
        self.output_dir = Path(output_dir).resolve()
        self.env_module = env_module
        self.num_workers = max(1, int(num_workers))

    @property
    def _eval_dir(self) -> Path:
        return self.output_dir / "evaluation"

    def _run(
        self,
        n_eval_rollouts: int,
        get_teammate_spec: Callable[[int], dict[str, Any]],
        rollout_dir_fn: Callable[[int], str],
    ) -> tuple[
        list[str | None],
        list[int],
        list[str | None],
        list[float | None],
        list[float | None],
        list[float | None],
        float,
    ]:
        return run_baseline_rollouts(
            env_module=self.env_module,
            n_eval_rollouts=n_eval_rollouts,
            eval_dir=self._eval_dir,
            eval_kwargs={},
            get_teammate_spec=get_teammate_spec,
            rollout_dir_fn=rollout_dir_fn,
            num_workers=self.num_workers,
        )

    def evaluate_eval_teammates(self, n_eval_rollouts: int = 50) -> dict:
        """Evaluate the null-action ego against standard heldout eval teammates.

        Writes ``baseline_eval_teammates.json``.
        """
        scripted_policies = _resolve_scripted_policies(self.env_module, "standard")
        teammate_entries: list[dict] = []
        for tm_idx, policy in enumerate(scripted_policies):
            policy_name = getattr(policy, "__name__", f"eval_teammate_{tm_idx}")
            (
                rollout_errors,
                traj_lengths,
                rollout_paths,
                per_rollout_team_rewards,
                per_rollout_ego_rewards,
                per_rollout_teammate_rewards,
                elapsed,
            ) = self._run(
                n_eval_rollouts=n_eval_rollouts,
                get_teammate_spec=lambda _i, _f=policy: {"kind": "scripted_factory", "factory": _f},
                rollout_dir_fn=lambda i, idx=tm_idx: f"baseline_rollout_{i}_with_eval_teammate_{idx}",
            )
            entry = _build_rollout_summary(
                n_eval_rollouts,
                rollout_errors,
                traj_lengths,
                rollout_paths,
                per_rollout_team_rewards,
                per_rollout_ego_rewards,
                per_rollout_teammate_rewards,
                elapsed,
            )
            entry.update({"idx": tm_idx, "name": str(policy_name)})
            teammate_entries.append(entry)

        return _save_summary(self._eval_dir, "baseline_eval_teammates.json", _build_teammate_eval_summary(n_eval_rollouts, teammate_entries))

    def evaluate_training_teammates(
        self,
        results_dir: str | Path,
        n_eval_rollouts: int = 50,
    ) -> dict:
        """Evaluate the null-action ego against a policy-code / reward-fn population.

        Writes ``baseline_training_teammates.json``.
        """
        refs = select_policy_refs_from_behavior_results(results_dir)
        teammate_entries: list[dict] = []
        for tm_idx, ref in enumerate(refs):
            (
                rollout_errors,
                traj_lengths,
                rollout_paths,
                per_rollout_team_rewards,
                per_rollout_ego_rewards,
                per_rollout_teammate_rewards,
                elapsed,
            ) = self._run(
                n_eval_rollouts=n_eval_rollouts,
                get_teammate_spec=lambda _i, _ref=ref: {"kind": "policy_ref", "ref": _ref},
                rollout_dir_fn=lambda i, idx=tm_idx: f"baseline_rollout_{i}_with_teammate_{idx}",
            )
            entry = _build_rollout_summary(
                n_eval_rollouts,
                rollout_errors,
                traj_lengths,
                rollout_paths,
                per_rollout_team_rewards,
                per_rollout_ego_rewards,
                per_rollout_teammate_rewards,
                elapsed,
            )
            entry["idx"] = tm_idx
            teammate_entries.append(entry)

        return _save_summary(self._eval_dir, "baseline_training_teammates.json", _build_teammate_eval_summary(n_eval_rollouts, teammate_entries))
