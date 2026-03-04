#!/usr/bin/env python3
"""Replay and re-evaluate runs from an existing experiment directory.

Example:
    python scripts/eval_results.py \
      experiments/b2t/results/ablation_neg_bestlast_fb/lbf/policy_code__gemini-3-flash-preview \
      --base-seed 42 \
      --num-rollouts 8
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

from popgen.utils.warnings import configure_warning_filters
from pqdm.processes import pqdm

from popgen.population.evaluation import BehaviorAdherenceEvaluator
from popgen.population.pipeline import _rollout_worker
from popgen.population.types import Behavior
from popgen.utils.config import LitellmConfig

configure_warning_filters(force=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


GENERATION_TO_POLICY_TYPE = {
    "policy_code": "PolicyCode",
    "rl_reward": "RewardFn",
    "llm_agent": "LLMAgent",
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_suffix_int(name: str, prefix: str) -> int:
    if not name.startswith(prefix):
        return -1
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return -1


def _latest_metadata_dir(result_path: Path) -> Path:
    metadata_root = result_path / "metadata"
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"metadata/ not found under {result_path}")

    dirs = sorted([d for d in metadata_root.iterdir() if d.is_dir() and (d / "summary.json").exists()], key=lambda d: d.name)
    if not dirs:
        raise FileNotFoundError(f"No metadata run with summary.json found under {metadata_root}")
    return dirs[-1]


def _resolve_run_output_dir(result_path: Path, run: dict[str, Any]) -> Path:
    output_dir = Path(str(run.get("output_dir", "")))
    if output_dir.exists():
        return output_dir

    behavior_idx = int(run.get("behavior_idx", 0))
    run_timestamp = str(run.get("run_timestamp", ""))
    fallback = result_path / "agents" / f"behavior_{behavior_idx}" / run_timestamp
    return fallback


def _sorted_attempt_dirs(run_output_dir: Path) -> list[Path]:
    attempts = [d for d in run_output_dir.iterdir() if d.is_dir() and d.name.startswith("attempt_")]
    return sorted(attempts, key=lambda p: _parse_suffix_int(p.name, "attempt_"))


def _select_attempt_dir(run_output_dir: Path) -> tuple[Path, int, str]:
    """Select attempt with highest adherence; fallback to last attempt if needed.

    Returns:
        (attempt_dir, attempt_idx, selection_note)
    """
    attempts = _sorted_attempt_dirs(run_output_dir)
    if not attempts:
        raise FileNotFoundError(f"No attempt_* directories in {run_output_dir}")

    scored_attempts: list[tuple[float, int, Path]] = []
    for attempt_dir in attempts:
        attempt_idx = _parse_suffix_int(attempt_dir.name, "attempt_")
        eval_path = attempt_dir / "evaluation.json"
        if not eval_path.exists():
            continue
        try:
            eval_payload = _load_json(eval_path)
            adherence = float(eval_payload.get("overall_metrics", {}).get("adherence"))
            scored_attempts.append((adherence, attempt_idx, attempt_dir))
        except Exception:
            continue

    if scored_attempts:
        best = max(scored_attempts, key=lambda x: (x[0], x[1]))
        return best[2], best[1], f"best_adherence={best[0]:.6f}"

    last_attempt = attempts[-1]
    last_idx = _parse_suffix_int(last_attempt.name, "attempt_")
    return last_attempt, last_idx, "fallback_last_attempt_no_valid_evaluation"


def _select_attempt_dirs(run_output_dir: Path, selection: str) -> list[tuple[Path, int, str]]:
    """Select one or more attempts according to selection policy."""
    if selection == "best":
        return [_select_attempt_dir(run_output_dir)]

    attempts = _sorted_attempt_dirs(run_output_dir)
    if not attempts:
        raise FileNotFoundError(f"No attempt_* directories in {run_output_dir}")

    if selection == "last":
        last_attempt = attempts[-1]
        last_idx = _parse_suffix_int(last_attempt.name, "attempt_")
        return [(last_attempt, last_idx, "last_attempt")]

    if selection == "all":
        return [
            (attempt_dir, _parse_suffix_int(attempt_dir.name, "attempt_"), "all_attempts")
            for attempt_dir in attempts
        ]

    raise ValueError(f"Unsupported attempt selection: {selection}")


def _resolve_loading_kwargs(policy_type: str, attempt_dir: Path) -> dict[str, Any]:
    if policy_type == "PolicyCode":
        path = attempt_dir / "policy.py"
        if not path.exists():
            raise FileNotFoundError(f"Missing policy file: {path}")
        return {"file_path": str(path)}
    if policy_type == "RewardFn":
        # RewardFn loader expects a directory containing checkpoints.
        return {"save_dir": str(attempt_dir)}
    if policy_type == "LLMAgent":
        path = attempt_dir / "llm_agent_config.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing LLM config file: {path}")
        return {"config_path": str(path)}
    raise ValueError(f"Unsupported policy type: {policy_type}")


def _seed_schedule(base_seed: int, k: int) -> list[int]:
    return [int(base_seed) + i * 1000 for i in range(k)]


def _run_rollouts_for_attempt(
    env_module: Any,
    policy_type: str,
    loading_kwargs: dict[str, Any],
    out_dir: Path,
    seeds: list[int],
    workers: int,
) -> tuple[list[dict[str, Any] | Exception], list[float]]:
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    rollout_args: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        rollout_args.append(
            {
                "rollout_fn": env_module.rollout_fn,
                "policy_type": policy_type,
                "create_env_fn": env_module.create_env,
                "scripted_policy_fn": env_module.ScriptedPolicy,
                "loading_kwargs": loading_kwargs,
                "rollout_subfolder": str(rollout_dir / f"rollout_{i}"),
                "rollout_kwargs": {"seed": int(seed)},
            }
        )

    raw_results = pqdm(
        rollout_args,
        _rollout_worker,
        n_jobs=max(1, int(workers)),
        desc="Re-eval rollouts",
    )

    rollouts: list[dict[str, Any] | Exception] = []
    latencies: list[float] = []
    for item in raw_results:
        if isinstance(item, Exception):
            rollouts.append(item)
            latencies.append(0.0)
            continue
        rollout_payload, elapsed = item
        rollouts.append(rollout_payload)
        latencies.append(float(elapsed))

    return rollouts, latencies


def _build_eval_inputs(
    behavior: Behavior,
    environment_description: str,
    rollouts: list[dict[str, Any] | Exception],
) -> list[dict[str, Any]]:
    inputs = []
    for result in rollouts:
        rollout_states: Any
        rollout_images: Any
        if isinstance(result, Exception):
            rollout_states = f"[ERROR] Rollout: {result}"
            rollout_images = None
        elif result.get("error") is not None:
            rollout_states = f"[ERROR] Rollout: {result['error']}"
            rollout_images = None
        else:
            rollout_states = result["states"]
            rollout_images = result["images"]

        inputs.append(
            {
                "behavior": str(behavior),
                "environment_description": environment_description,
                "rollout_states": rollout_states,
                "rollout_images": rollout_images,
            }
        )
    return inputs


def _resolve_judge_model(judge_model_override: str | None, selected_attempt_dir: Path) -> str | None:
    if judge_model_override:
        return judge_model_override

    prev_eval = selected_attempt_dir / "evaluation.json"
    if not prev_eval.exists():
        return None

    try:
        payload = _load_json(prev_eval)
        return payload.get("metadata", {}).get("judge_model")
    except Exception:
        return None


def _resolve_output_root(result_path: Path, output_dir: str | None, skip_existing: bool) -> Path:
    if output_dir:
        return Path(output_dir).resolve()

    eval_root = result_path / "eval"
    if skip_existing and eval_root.is_dir():
        existing_eval_dirs = sorted([d for d in eval_root.iterdir() if d.is_dir()], key=lambda d: d.name)
        if existing_eval_dirs:
            return existing_eval_dirs[-1]

    return eval_root / _timestamp()


def _load_existing_attempt_eval(out_attempt_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return prior successful (run_eval_summary, evaluation) payloads if present."""
    summary_path = out_attempt_dir / "run_eval_summary.json"
    evaluation_path = out_attempt_dir / "evaluation.json"
    if not summary_path.exists() or not evaluation_path.exists():
        return None

    try:
        run_summary = _load_json(summary_path)
        evaluation_payload = _load_json(evaluation_path)
    except Exception:
        return None

    if run_summary.get("status") != "ok":
        return None
    return run_summary, evaluation_payload


def _write_runs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "behavior_idx",
        "trial_id",
        "run_timestamp",
        "generation_type",
        "policy_type",
        "attempt_selection",
        "status",
        "selection_note",
        "selected_attempt_idx",
        "adherence",
        "adjusted_adherence",
        "n_evaluated",
        "n_errors",
        "judge_model",
        "base_seed",
        "num_rollouts",
        "seeds",
        "source_run_dir",
        "source_attempt_dir",
        "output_dir",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and re-evaluate an existing experiment output directory.")
    parser.add_argument("result_path", type=str, help="Single experiment directory (e.g. .../lbf/policy_code__gemini-3-flash-preview)")
    parser.add_argument("--base-seed", type=int, required=True, help="Base seed used for rollout seeds: base_seed + i*1000")
    parser.add_argument("-k", "--num-rollouts", type=int, required=True, help="Number of rollouts per selected run")
    parser.add_argument(
        "--attempt-selection",
        choices=["best", "all", "last"],
        default="best",
        help="Which attempt(s) to evaluate per run (default: best).",
    )
    parser.add_argument("--judge-model", type=str, default=None, help="Optional adherence judge model override")
    parser.add_argument(
        "--judge-timeout-sec",
        type=float,
        default=120.0,
        help="Per-request timeout (seconds) for judge model calls (default: 120).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory. Default: <result_path>/eval/<timestamp>")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip attempt re-evaluation if output attempt already has successful run_eval_summary.json and evaluation.json.",
    )
    parser.add_argument("--workers", type=int, default=5, help="Rollout and evaluator worker count")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of runs processed")
    parser.add_argument("--dry-run", action="store_true", help="Discover and select attempts only; do not run rollouts/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_path).resolve()
    if not result_path.exists():
        raise FileNotFoundError(f"Result path does not exist: {result_path}")

    metadata_dir = _latest_metadata_dir(result_path)
    summary_path = metadata_dir / "summary.json"
    behaviors_path = metadata_dir / "behaviors.json"

    if not behaviors_path.exists():
        raise FileNotFoundError(f"Missing behaviors file in selected metadata dir: {behaviors_path}")

    summary = _load_json(summary_path)
    behaviors_data = _load_json(behaviors_path)
    behaviors = [Behavior.model_validate(b) for b in behaviors_data]

    runs = list(summary.get("runs", []))
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]

    env_name = str(summary.get("env"))
    if not env_name:
        raise ValueError(f"Could not resolve environment from {summary_path}")
    env_module = importlib.import_module(f"popgen.envs.{env_name}")

    output_root = _resolve_output_root(result_path, args.output_dir, args.skip_existing)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("RE-EVAL RUNNER")
    logging.info("=" * 60)
    logging.info(f"Experiment: {result_path}")
    logging.info(f"Metadata:   {metadata_dir.name}")
    logging.info(f"Env:        {env_name}")
    logging.info(f"Runs:       {len(runs)}")
    logging.info(f"Rollouts:   {args.num_rollouts} per run")
    logging.info(f"Attempts:   {args.attempt_selection}")
    logging.info(f"Judge timeout: {args.judge_timeout_sec}s")
    logging.info(f"Base seed:  {args.base_seed}")
    logging.info(f"Dry run:    {args.dry_run}")
    logging.info(f"Skip existing: {args.skip_existing}")
    logging.info(f"Output:     {output_root}")

    rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    total_start = time()

    for run in runs:
        behavior_idx = int(run.get("behavior_idx", 0))
        trial_idx = int(run.get("trial_idx", 0))
        run_timestamp = str(run.get("run_timestamp", "unknown"))
        trial_id = run_timestamp if run_timestamp and run_timestamp != "unknown" else f"trial_{trial_idx}"
        generation_type = str(run.get("generation_type", summary.get("generation_type", "")))
        policy_type = GENERATION_TO_POLICY_TYPE.get(generation_type)

        out_run_dir = (
            output_root
            / "runs"
            / f"behavior_{behavior_idx}"
            / f"trial_{trial_id}"
        )
        out_run_dir.mkdir(parents=True, exist_ok=True)

        row_template: dict[str, Any] = {
            "behavior_idx": behavior_idx,
            "trial_id": trial_id,
            "run_timestamp": run_timestamp,
            "generation_type": generation_type,
            "policy_type": policy_type or "",
            "attempt_selection": args.attempt_selection,
            "status": "failed",
            "selection_note": "",
            "selected_attempt_idx": None,
            "adherence": None,
            "adjusted_adherence": None,
            "n_evaluated": None,
            "n_errors": None,
            "judge_model": None,
            "base_seed": int(args.base_seed),
            "num_rollouts": int(args.num_rollouts),
            "seeds": json.dumps(_seed_schedule(args.base_seed, args.num_rollouts)),
            "source_run_dir": "",
            "source_attempt_dir": "",
            "output_dir": str(out_run_dir),
            "error": "",
        }

        try:
            if policy_type is None:
                raise ValueError(f"Unsupported generation_type '{generation_type}'")
            source_run_dir = _resolve_run_output_dir(result_path, run)
            if not source_run_dir.exists():
                raise FileNotFoundError(f"Run output directory not found: {source_run_dir}")
            selected_attempts = _select_attempt_dirs(source_run_dir, args.attempt_selection)
        except Exception as e:
            row = {**row_template}
            row["status"] = "failed"
            row["error"] = str(e)
            rows.append(row)
            run_records.append({**row, "status": "failed"})
            _write_json(out_run_dir / "run_eval_summary.json", {"status": "failed", "error": str(e), **row})
            continue

        for selected_attempt_dir, selected_attempt_idx, selection_note in selected_attempts:
            out_attempt_dir = out_run_dir / f"attempt_{selected_attempt_idx}"
            out_attempt_dir.mkdir(parents=True, exist_ok=True)

            row: dict[str, Any] = {
                **row_template,
                "selection_note": selection_note,
                "selected_attempt_idx": selected_attempt_idx,
                "source_run_dir": str(source_run_dir),
                "source_attempt_dir": str(selected_attempt_dir),
                "output_dir": str(out_attempt_dir),
            }

            try:
                if args.skip_existing:
                    existing = _load_existing_attempt_eval(out_attempt_dir)
                    if existing is not None:
                        existing_summary, existing_eval = existing
                        existing_metrics = existing_eval.get("overall_metrics", {})
                        row["status"] = "skipped_existing"
                        row["judge_model"] = (
                            str(existing_summary.get("judge_model"))
                            if existing_summary.get("judge_model") is not None
                            else existing_eval.get("metadata", {}).get("judge_model")
                        )
                        row["adherence"] = _safe_float(existing_metrics.get("adherence"), 0.0)
                        row["adjusted_adherence"] = _safe_float(existing_metrics.get("adjusted_adherence"), 0.0)
                        row["n_evaluated"] = _safe_int(existing_metrics.get("n_evaluated"), 0)
                        row["n_errors"] = _safe_int(existing_metrics.get("n_errors"), 0)
                        run_records.append(existing_summary)
                        rows.append(row)
                        continue

                if args.dry_run:
                    row["status"] = "planned"
                    run_records.append({**row, "dry_run": True})
                    rows.append(row)
                    _write_json(out_attempt_dir / "run_eval_summary.json", row)
                    continue

                judge_model = _resolve_judge_model(args.judge_model, selected_attempt_dir)
                if not judge_model:
                    raise ValueError(
                        "No judge model resolved. Provide --judge-model or ensure source evaluation.json has metadata.judge_model."
                    )
                row["judge_model"] = judge_model

                loading_kwargs = _resolve_loading_kwargs(policy_type, selected_attempt_dir)
                seeds = _seed_schedule(args.base_seed, args.num_rollouts)

                rollout_start = time()
                rollouts, rollout_latencies = _run_rollouts_for_attempt(
                    env_module=env_module,
                    policy_type=policy_type,
                    loading_kwargs=loading_kwargs,
                    out_dir=out_attempt_dir,
                    seeds=seeds,
                    workers=args.workers,
                )
                rollout_elapsed = time() - rollout_start

                if behavior_idx >= len(behaviors):
                    raise IndexError(
                        f"Behavior index {behavior_idx} out of range for behaviors file of length {len(behaviors)}"
                    )
                behavior = behaviors[behavior_idx]
                eval_inputs = _build_eval_inputs(
                    behavior=behavior,
                    environment_description=env_module.ENVIRONMENT_DESCRIPTION,
                    rollouts=rollouts,
                )

                evaluator = BehaviorAdherenceEvaluator(
                    model_config=LitellmConfig(
                        model_name=judge_model,
                        generation_kwargs={"timeout": float(args.judge_timeout_sec)},
                    ),
                    num_workers=max(1, int(args.workers)),
                )

                eval_start = time()
                eval_result = evaluator.evaluate(eval_inputs)
                eval_elapsed = time() - eval_start

                for i, metric in enumerate(eval_result.instance_metrics):
                    metric["rollout_data_path"] = os.path.abspath(
                        str(out_attempt_dir / "rollouts" / f"rollout_{i}" / "rollout_data.jsonl")
                    )

                evaluation_payload = {
                    "instance_metrics": eval_result.instance_metrics,
                    "overall_metrics": eval_result.overall_metrics,
                    "metadata": eval_result.metadata,
                }
                _write_json(out_attempt_dir / "evaluation.json", evaluation_payload)

                run_eval_summary = {
                    "status": "ok",
                    "behavior_idx": behavior_idx,
                    "trial_id": trial_id,
                    "run_timestamp": run_timestamp,
                    "generation_type": generation_type,
                    "policy_type": policy_type,
                    "attempt_selection": args.attempt_selection,
                    "selection_note": selection_note,
                    "selected_attempt_idx": selected_attempt_idx,
                    "source_run_dir": str(source_run_dir),
                    "source_attempt_dir": str(selected_attempt_dir),
                    "output_attempt_dir": str(out_attempt_dir),
                    "base_seed": int(args.base_seed),
                    "seeds": seeds,
                    "num_rollouts": int(args.num_rollouts),
                    "judge_model": judge_model,
                    "latencies": {
                        "rollout_total": rollout_elapsed,
                        "rollout_per_seed": rollout_latencies,
                        "evaluation": eval_elapsed,
                    },
                    "overall_metrics": eval_result.overall_metrics,
                }
                _write_json(out_attempt_dir / "run_eval_summary.json", run_eval_summary)

                row["status"] = "ok"
                row["adherence"] = _safe_float(eval_result.overall_metrics.get("adherence"), 0.0)
                row["adjusted_adherence"] = _safe_float(eval_result.overall_metrics.get("adjusted_adherence"), 0.0)
                row["n_evaluated"] = _safe_int(eval_result.overall_metrics.get("n_evaluated"), 0)
                row["n_errors"] = _safe_int(eval_result.overall_metrics.get("n_errors"), 0)
                run_records.append(run_eval_summary)
            except Exception as e:
                row["status"] = "failed"
                row["error"] = str(e)
                run_records.append({**row, "status": "failed"})
                _write_json(out_attempt_dir / "run_eval_summary.json", {"status": "failed", "error": str(e), **row})

            rows.append(row)

    successful_rows = [r for r in rows if r["status"] in {"ok", "skipped_existing"}]
    skipped_existing_rows = [r for r in rows if r["status"] == "skipped_existing"]
    planned_rows = [r for r in rows if r["status"] == "planned"]
    counted_success_rows = successful_rows + planned_rows if args.dry_run else successful_rows
    run_key = lambda r: (int(r["behavior_idx"]), str(r["trial_id"]), str(r["run_timestamp"]))
    run_keys = {run_key(r) for r in rows}
    run_success_keys = {run_key(r) for r in counted_success_rows}
    run_skipped_existing_keys = {run_key(r) for r in skipped_existing_rows}
    run_planned_keys = {run_key(r) for r in planned_rows}
    avg_adherence = (
        sum(float(r["adherence"]) for r in successful_rows if r["adherence"] is not None) / len(successful_rows)
        if successful_rows
        else 0.0
    )

    final_summary = {
        "source_experiment_path": str(result_path),
        "source_metadata_timestamp": metadata_dir.name,
        "env": env_name,
        "generation_type": summary.get("generation_type"),
        "generation_model": summary.get("generation_model"),
        "attempt_selection": args.attempt_selection,
        "base_seed": int(args.base_seed),
        "num_rollouts": int(args.num_rollouts),
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "num_runs_processed": len(run_keys),
        "num_attempt_evals_processed": len(rows),
        "num_runs_succeeded": len(run_success_keys),
        "num_attempt_evals_succeeded": len(counted_success_rows),
        "num_runs_skipped_existing": len(run_skipped_existing_keys),
        "num_attempt_evals_skipped_existing": len(skipped_existing_rows),
        "num_runs_planned": len(run_planned_keys),
        "num_attempt_evals_planned": len(planned_rows),
        "num_runs_failed": len(run_keys) - len(run_success_keys),
        "num_attempt_evals_failed": len(rows) - len(counted_success_rows),
        "average_adherence": avg_adherence,
        "total_time": time() - total_start,
        "runs": run_records,
    }

    _write_json(output_root / "summary.json", final_summary)
    _write_runs_csv(output_root / "runs.csv", rows)

    logging.info("=" * 60)
    logging.info("RE-EVAL COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Succeeded: {len(counted_success_rows)}/{len(rows)}")
    logging.info(f"Average adherence: {avg_adherence:.4f}")
    logging.info(f"Summary: {output_root / 'summary.json'}")
    logging.info(f"CSV:     {output_root / 'runs.csv'}")


if __name__ == "__main__":
    main()
