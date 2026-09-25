#!/usr/bin/env python3
"""Gather experiment results into a CSV file.

Usage:
    python gather_results.py results/lbf/policy_code/VerbalizedSamplingStrategy [dir2 ...] -o results.csv

Each input directory is treated as one experiment.

Supported layouts:
1) New layout:
    <experiment_dir>/
        agents/
            behavior_<i>/<run_timestamp>/attempt_<j>/
                evaluation.json  latency.json  model_outputs.jsonl
        metadata/
            <metadata_timestamp>/behaviors.json
            <metadata_timestamp>/summary.json

2) Legacy layout:
    <experiment_dir>/<run_id>/                      # .hydra/config.yaml lives here
        behavior_<i>/attempt_<j>/                   # when n_trials == 1
            evaluation.json  latency.json  model_outputs.jsonl
        trial_<k>/behavior_<i>/attempt_<j>/FILES    # when n_trials > 1
"""

import argparse
import csv
import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
import yaml

_bpe_encoder = None
_warned_tokenizer_fallback = False


def _ensure_tokenizer_loaded() -> None:
    global _bpe_encoder, _warned_tokenizer_fallback
    if _bpe_encoder is not None:
        return

    try:
        # Use the latest tiktoken base vocabulary.
        _bpe_encoder = tiktoken.get_encoding("o200k_base")
        return
    except Exception:
        # Final fallback keeps script runnable in fully offline environments.
        if not _warned_tokenizer_fallback:
            warnings.warn(
                "tiktoken tokenizer files not available locally; using approximate token counting fallback.",
                RuntimeWarning,
            )
            _warned_tokenizer_fallback = True


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    _ensure_tokenizer_loaded()
    if _bpe_encoder is not None:
        return len(_bpe_encoder.encode(text))
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _parse_idx(folder_name: str) -> int:
    """Extract trailing numeric index from names like behavior_3, trial_0."""
    m = re.search(r"_(\d+)$", folder_name)
    return int(m.group(1)) if m else 0


def _sorted_dirs(parent: Path, prefix: str) -> list[Path]:
    return sorted(
        [d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: _parse_idx(d.name),
    )


def _is_new_layout(experiment_dir: Path) -> bool:
    return (experiment_dir / "agents").is_dir() and (experiment_dir / "metadata").is_dir()


def _is_legacy_layout(experiment_dir: Path) -> bool:
    """Return True if any immediate child looks like a legacy hydra run directory.

    A child is a legacy run dir only if it has .hydra/config.yaml but is NOT
    itself a new-layout experiment dir (which also carries .hydra/ from Hydra).
    """
    return any(
        not _is_new_layout(child) and (child / ".hydra" / "config.yaml").exists()
        for child in experiment_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )



def _infer_split(path: Path) -> str:
    """Infer dataset split from directory path."""
    return "eval" if "eval" in path.parts else "train"


def _normalize_split(split: str) -> str:
    """Normalize split labels for downstream analysis CSV."""
    mapping = {"train": "dev", "eval": "test"}
    return mapping.get(split, split)


def _gather_container(container_dir: Path) -> list[dict]:
    rows = []
    for child in sorted(container_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            rows.extend(gather_experiment(child))
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _metric_success(metric: dict[str, Any]) -> bool:
    """Normalize success for different evaluator schemas."""
    if "evaluation" in metric:
        return bool(metric.get("evaluation", False))
    adherence = metric.get("adherence")
    return bool(adherence is not None and float(adherence) >= 1.0)


def _normalize_generation_latency(value: Any) -> float | None:
    """Normalize generation latency to a scalar for CSV."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        numeric_values = [float(v) for v in value.values() if isinstance(v, (int, float))]
        return sum(numeric_values) if numeric_values else None
    return None


def _collect_text_fields(obj: Any) -> str:
    """Collect likely generation text recursively from a JSON-like object."""
    text_fields = []

    def _walk(value: Any):
        if value is None:
            return
        if isinstance(value, str):
            text_fields.append(value)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)
            return
        if isinstance(value, dict):
            # Prefer common generation payload keys to avoid counting metadata.
            for key in ("reasoning", "code", "output", "response", "text", "completion", "content", "message"):
                if key in value:
                    _walk(value[key])

    _walk(obj)
    return "\n".join(text_fields)


def _collect_text_for_key(obj: Any, key_name: str) -> str:
    """Collect text recursively for values attached to a specific key."""
    text_fields = []

    def _collect_all_text(value: Any):
        if value is None:
            return
        if isinstance(value, str):
            text_fields.append(value)
            return
        if isinstance(value, list):
            for item in value:
                _collect_all_text(item)
            return
        if isinstance(value, dict):
            for nested_value in value.values():
                _collect_all_text(nested_value)

    def _walk(value: Any):
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)
            return
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if key == key_name:
                    _collect_all_text(nested_value)
                else:
                    _walk(nested_value)

    _walk(obj)
    return "\n".join(text_fields)


def _select_generation_output(
    model_outputs: list[dict], rollout_idx: int, generation_type: str, n_rollouts: int
) -> dict[str, Any]:
    """Select model output entry corresponding to a given rollout index.

    Uses the last model output entry for every rollout: the file records one
    entry per *generation*, not per rollout, so the final entry is the one the
    rollouts were actually produced from.
    """
    if not model_outputs:
        return {}

    return model_outputs[-1]


def _generation_tokens(
    model_outputs: list[dict], rollout_idx: int, generation_type: str, n_rollouts: int
) -> tuple[int, int]:
    """Compute (reasoning_tokens, code_tokens) for a given rollout index."""
    entry = _select_generation_output(
        model_outputs=model_outputs,
        rollout_idx=rollout_idx,
        generation_type=generation_type,
        n_rollouts=n_rollouts,
    )
    if not entry:
        return 0, 0

    reasoning_text = _collect_text_for_key(entry, "reasoning")
    code_text = _collect_text_for_key(entry, "code")

    # Fallback for schemas that do not expose explicit "reasoning"/"code" keys.
    if not reasoning_text and not code_text:
        return 0, _count_tokens(_collect_text_fields(entry))

    return _count_tokens(reasoning_text), _count_tokens(code_text)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _collect_attempt_rows(
    *,
    experiment_id: str,
    run_name: str,
    run_id: str,
    split: str,
    env: str,
    seed: int,
    generation_type: str,
    generation_model: str,
    trial_id: str,
    behavior_id: int,
    run_output_dir: Path,
) -> list[dict]:
    return _collect_rows_for_attempt_dirs(
        experiment_id=experiment_id,
        run_name=run_name,
        run_id=run_id,
        split=split,
        env=env,
        seed=seed,
        generation_type=generation_type,
        generation_model=generation_model,
        trial_id=trial_id,
        behavior_id=behavior_id,
        attempt_dirs=_sorted_dirs(run_output_dir, "attempt_"),
    )


def _collect_single_attempt_rows(
    *,
    experiment_id: str,
    run_name: str,
    run_id: str,
    split: str,
    env: str,
    seed: int,
    generation_type: str,
    generation_model: str,
    trial_id: str,
    behavior_id: int,
    attempt_dir: Path,
) -> list[dict]:
    return _collect_rows_for_attempt_dirs(
        experiment_id=experiment_id,
        run_name=run_name,
        run_id=run_id,
        split=split,
        env=env,
        seed=seed,
        generation_type=generation_type,
        generation_model=generation_model,
        trial_id=trial_id,
        behavior_id=behavior_id,
        attempt_dirs=[attempt_dir],
    )


def _collect_rows_for_attempt_dirs(
    *,
    experiment_id: str,
    run_name: str,
    run_id: str,
    split: str,
    env: str,
    seed: int,
    generation_type: str,
    generation_model: str,
    trial_id: str,
    behavior_id: int,
    attempt_dirs: list[Path],
) -> list[dict]:
    rows = []
    for attempt_dir in attempt_dirs:
        if not attempt_dir.exists() or not attempt_dir.is_dir():
            continue
        attempt_id = _parse_idx(attempt_dir.name)

        eval_path = attempt_dir / "evaluation.json"
        if not eval_path.exists():
            continue
        eval_data = _load_json(eval_path)
        instance_metrics = eval_data.get("instance_metrics", [])

        latency_path = attempt_dir / "latency.json"
        latency_data: dict = _load_json(latency_path) if latency_path.exists() else {}
        generation_latency = _normalize_generation_latency(latency_data.get("generation"))
        rollout_latency_raw = latency_data.get("rollout", [])
        rollout_latencies = rollout_latency_raw if isinstance(rollout_latency_raw, list) else []

        model_outputs = _read_jsonl(attempt_dir / "model_outputs.jsonl")

        for rollout_idx, metric in enumerate(instance_metrics):
            rollout_latency = rollout_latencies[rollout_idx] if rollout_idx < len(rollout_latencies) else None
            reasoning_tokens, code_tokens = _generation_tokens(
                model_outputs=model_outputs,
                rollout_idx=rollout_idx,
                generation_type=generation_type,
                n_rollouts=len(instance_metrics),
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "run_name": run_name,
                    "run_id": run_id,
                    "split": _normalize_split(split),
                    "env": env,
                    "seed": seed,
                    "generation_type": generation_type,
                    "generation_model": generation_model,
                    "trial_id": trial_id,
                    "behavior_id": behavior_id,
                    "attempt_id": attempt_id,
                    "rollout_idx": rollout_idx,
                    "success": _metric_success(metric),
                    "reasoning_tokens": reasoning_tokens,
                    "code_tokens": code_tokens,
                    "generation_tokens": reasoning_tokens + code_tokens,
                    "generation_latency": generation_latency,
                    "rollout_latency": rollout_latency,
                }
            )
    return rows


def _gather_experiment_eval_layout(experiment_dir: Path) -> list[dict]:
    """Gather eval results by scanning evaluation.json files on disk."""
    experiment_id = experiment_dir.name
    run_name = experiment_dir.name
    rows = []

    eval_root = experiment_dir / "eval"
    if not eval_root.is_dir():
        return rows

    # Read experiment-level metadata for env/generation info
    metadata_root = experiment_dir / "metadata"
    env = ""
    generation_type = ""
    generation_model = experiment_dir.name
    seed = 0
    if metadata_root.is_dir():
        meta_dirs = sorted([d for d in metadata_root.iterdir() if d.is_dir()], key=lambda d: d.name)
        if meta_dirs:
            summary_path = meta_dirs[-1] / "summary.json"
            if summary_path.exists():
                meta = _load_json(summary_path)
                env = meta.get("env", "")
                generation_type = meta.get("generation_type", "")
                generation_model = meta.get("generation_model", experiment_dir.name)

    # Scan all evaluation.json files across all eval dirs
    for eval_path in sorted(eval_root.glob("**/evaluation.json")):
        # Path: eval/<ts>/runs/behavior_{idx}/trial_{id}/attempt_{idx}/evaluation.json
        attempt_dir = eval_path.parent
        trial_dir = attempt_dir.parent
        behavior_dir = trial_dir.parent
        eval_dir = behavior_dir.parent.parent  # up past "runs"

        behavior_id = _parse_idx(behavior_dir.name)
        trial_id = trial_dir.name.split("_", 1)[1]
        run_group_id = eval_dir.name

        rows.extend(
            _collect_single_attempt_rows(
                experiment_id=experiment_id,
                run_name=run_name,
                run_id=run_group_id,
                split="eval",
                env=env,
                seed=seed,
                generation_type=generation_type,
                generation_model=generation_model,
                trial_id=trial_id,
                behavior_id=behavior_id,
                attempt_dir=attempt_dir,
            )
        )
    return rows


def _gather_experiment_new_layout(experiment_dir: Path) -> list[dict]:
    experiment_id = experiment_dir.name
    run_name = experiment_dir.name
    split = _infer_split(experiment_dir)
    rows = []

    metadata_root = experiment_dir / "metadata"
    metadata_runs = sorted([d for d in metadata_root.iterdir() if d.is_dir()], key=lambda d: d.name)
    for metadata_dir in metadata_runs:
        summary_path = metadata_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary_data = _load_json(summary_path)

        env = summary_data.get("env", "")
        generation_type = summary_data.get("generation_type", "")
        generation_model = summary_data.get("generation_model", experiment_dir.name)
        run_group_id = metadata_dir.name

        for run in summary_data.get("runs", []):
            behavior_id = int(run.get("behavior_idx", 0))
            run_timestamp = run.get("run_timestamp", "")
            trial_idx = int(run.get("trial_idx", 0))
            trial_id = str(run_timestamp) if run_timestamp else f"trial_{trial_idx}"

            run_output_dir = Path(run.get("output_dir", ""))
            if not run_output_dir.exists():
                run_output_dir = experiment_dir / "agents" / f"behavior_{behavior_id}" / run_timestamp
            if not run_output_dir.exists():
                continue

            run_meta_path = run_output_dir / "run_metadata.json"
            run_meta = _load_json(run_meta_path) if run_meta_path.exists() else {}

            seed = int(run.get("seed", run_meta.get("seed", 0)))
            run_generation_type = run.get("generation_type", run_meta.get("generation_type", generation_type))
            rows.extend(
                _collect_attempt_rows(
                    experiment_id=experiment_id,
                    run_name=run_name,
                    run_id=run_group_id,
                    split=split,
                    env=env,
                    seed=seed,
                    generation_type=run_generation_type,
                    generation_model=generation_model,
                    trial_id=trial_id,
                    behavior_id=behavior_id,
                    run_output_dir=run_output_dir,
                )
            )

    rows.extend(_gather_experiment_eval_layout(experiment_dir))
    return rows


def _gather_experiment_legacy_layout(experiment_dir: Path) -> list[dict]:
    experiment_id = experiment_dir.name
    run_name = experiment_dir.name
    split = _infer_split(experiment_dir)
    rows = []

    run_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and not d.name.startswith(".")], key=lambda d: d.name)
    for run_dir in run_dirs:
        run_id = run_dir.name
        hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
        if not hydra_cfg_path.exists():
            continue

        with open(hydra_cfg_path) as f:
            hydra_cfg = yaml.safe_load(f)

        env = hydra_cfg.get("env", "")
        seed = hydra_cfg.get("seed", 0)
        generation_cfg = hydra_cfg.get("generation", {})
        generation_type = generation_cfg.get("type", "")
        generation_model = generation_cfg.get("model_name", "")

        trial_dirs = _sorted_dirs(run_dir, "trial_")
        trials = [(d, d.name) for d in trial_dirs] if trial_dirs else [(run_dir, run_dir.name)]

        for trial_dir, trial_id in trials:
            for behavior_dir in _sorted_dirs(trial_dir, "behavior_"):
                behavior_id = _parse_idx(behavior_dir.name)
                rows.extend(
                    _collect_attempt_rows(
                        experiment_id=experiment_id,
                        run_name=run_name,
                        run_id=run_id,
                        split=split,
                        env=env,
                        seed=seed,
                        generation_type=generation_type,
                        generation_model=generation_model,
                        trial_id=trial_id,
                        behavior_id=behavior_id,
                        run_output_dir=behavior_dir,
                    )
                )
    return rows


def gather_experiment(experiment_dir: Path) -> list[dict]:
    if _is_new_layout(experiment_dir):
        return _gather_experiment_new_layout(experiment_dir)
    if _is_legacy_layout(experiment_dir):
        return _gather_experiment_legacy_layout(experiment_dir)
    return _gather_container(experiment_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Gather experiment results from one or more experiment directories into a CSV."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        metavar="DIR",
        help="Experiment result folder(s) to gather (one per experiment_id)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: results_<timestamp>.csv)",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for exp_dir in args.experiment_dirs:
        path = Path(exp_dir).resolve()
        if not path.is_dir():
            print(f"[WARN] Not a directory, skipping: {exp_dir}")
            continue
        print(f"Gathering: {path}")
        rows = gather_experiment(path)
        all_rows.extend(rows)
        print(f"  → {len(rows)} row(s)")

    if not all_rows:
        print("No data found.")
        return

    fieldnames = [
        "experiment_id",
        "run_name",
        "run_id",
        "split",
        "env",
        "seed",
        "generation_type",
        "generation_model",
        "trial_id",
        "behavior_id",
        "attempt_id",
        "rollout_idx",
        "success",
        "reasoning_tokens",
        "code_tokens",
        "generation_tokens",
        "generation_latency",
        "rollout_latency",
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output:
        p = Path(args.output)
        out_path = p.with_stem(f"{p.stem}__{timestamp}")
    else:
        out_path = Path(f"./data/compiled_results/results__{timestamp}.csv")
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} row(s) → {out_path.resolve()}")


if __name__ == "__main__":
    main()
