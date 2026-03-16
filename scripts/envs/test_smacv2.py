#!/usr/bin/env python3
"""Smoke-test rollout and training entry points for SMACv2.

Artifacts are saved under scripts/smacv2_test by default.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from popgen.envs.smacv2 import utils as smac_utils
from popgen.envs.smacv2.utils import rollout_smacv2_env, train_smacv2_env
from popgen.utils.logging import save_rollout


def _safe_close(env: Any) -> None:
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def _policy_from_experiment(experiment: Any):
    if hasattr(experiment, "group_policies"):
        if "ego" in experiment.group_policies:
            return experiment.group_policies["ego"]
        if "agents" in experiment.group_policies:
            return experiment.group_policies["agents"]
    raise RuntimeError("Could not find trained policy group in experiment.group_policies")


def _get_obs_feature_names(env) -> list[str]:
    try:
        if hasattr(env, "_env") and hasattr(env._env, "get_obs_feature_names"):
            return list(env._env.get_obs_feature_names())
        if hasattr(env, "get_obs_feature_names"):
            return list(env.get_obs_feature_names())
    except Exception:
        pass
    return []


def _print_raw_observations(states: list[dict[str, Any]], label: str, feature_names: list[str] | None = None) -> None:
    print(f"\n=== Raw observations: {label} ===")
    feature_names = feature_names or []
    for step_idx, state in enumerate(states):
        # Format: {"ego_team": {"ego_agent": {"state": [...], ...}, ...}, "opponent_team": {...}}
        all_agents: dict[str, list] = {}
        for team_key in ("ego_team", "opponent_team"):
            for agent_name, agent_data in state.get(team_key, {}).items():
                all_agents[f"{team_key}/{agent_name}"] = agent_data.get("state", [])

        if not all_agents:
            print(f"[step {step_idx}] no agent observations found in rollout state")
            continue

        print(f"[step {step_idx}] n_agents={len(all_agents)}")
        for agent_label, obs_vec in all_agents.items():
            print(f"  {agent_label}: {json.dumps(obs_vec)}")
            if feature_names and len(feature_names) == len(obs_vec):
                for feat_name, feat_val in zip(feature_names, obs_vec):
                    print(f"    {feat_name}: {feat_val}")
            elif feature_names:
                print(
                    f"    [warning] feature-name length ({len(feature_names)}) "
                    f"!= obs length ({len(obs_vec)}), skipping name-value mapping"
                )


def _run_common_rollout(output_dir: Path, max_steps: int, seed: int) -> dict[str, Any]:
    env = None
    try:
        env = smac_utils.create_env(task_name="terran_2_vs_4", num_envs=1, seed=seed, device="cpu")
        feature_names = _get_obs_feature_names(env)
        policy = smac_utils.ScriptedPolicy(env)
        states, images, error = rollout_smacv2_env(
            policy=policy,
            env=env,
            max_steps=max_steps,
            seed=seed,
        )
        _print_raw_observations(
            states,
            label="smacv2_common.rollout_smacv2_env",
            feature_names=feature_names,
        )
        save_rollout(states, images or [], str(output_dir))
        return {
            "status": "ok" if error is None else "error",
            "error": None if error is None else str(error),
            "n_states": len(states or []),
            "n_images": len(images or []),
            "path": str(output_dir),
        }
    finally:
        _safe_close(env)


def _run_utils_rollout(output_dir: Path, max_steps: int, seed: int) -> dict[str, Any]:
    env = None
    try:
        env = smac_utils.create_env(task_name="terran_2_vs_4", num_envs=1, seed=seed, device="cpu")
        feature_names = _get_obs_feature_names(env)
        policy = smac_utils.ScriptedPolicy(env)
        states, images, error = smac_utils.rollout_fn(
            policy=policy,
            env=env,
            max_steps=max_steps,
            seed=seed,
        )
        _print_raw_observations(states, label="smacv2.utils.rollout_fn", feature_names=feature_names)
        save_rollout(states, images or [], str(output_dir))
        return {
            "status": "ok" if error is None else "error",
            "error": None if error is None else str(error),
            "n_states": len(states or []),
            "n_images": len(images or []),
            "path": str(output_dir),
        }
    finally:
        _safe_close(env)


def _run_common_train_and_rollout(output_dir: Path, max_steps: int, seed: int) -> dict[str, Any]:
    train_dir = output_dir / "train"
    rollout_dir = output_dir / "train_rollout"
    train_dir.mkdir(parents=True, exist_ok=True)

    experiment = train_smacv2_env(
        reward_fn=None,
        output_dir=str(train_dir),
        run_name="smacv2_common_test",
        smacv2_task="TERRAN_2_VS_4",
        task_name="terran_2_vs_4",
        rl_cfg_dir=smac_utils.RL_CFG_DIR,
    )

    policy = _policy_from_experiment(experiment)
    env = getattr(experiment, "test_env", None)
    try:
        if env is None:
            env = smac_utils.create_env(task_name="terran_2_vs_4", num_envs=1, seed=seed, device="cpu")
        feature_names = _get_obs_feature_names(env)

        states, images, error = rollout_smacv2_env(
            policy=policy,
            env=env,
            max_steps=max_steps,
            seed=seed,
        )
        _print_raw_observations(
            states,
            label="smacv2_common.train_smacv2_env + rollout_smacv2_env",
            feature_names=feature_names,
        )
        save_rollout(states, images or [], str(rollout_dir))

        return {
            "status": "ok" if error is None else "error",
            "error": None if error is None else str(error),
            "n_states": len(states or []),
            "n_images": len(images or []),
            "train_path": str(train_dir),
            "rollout_path": str(rollout_dir),
        }
    finally:
        _safe_close(env)


def _run_utils_train_and_rollout(output_dir: Path, max_steps: int, seed: int) -> dict[str, Any]:
    train_dir = output_dir / "train"
    rollout_dir = output_dir / "train_rollout"
    train_dir.mkdir(parents=True, exist_ok=True)

    experiment = smac_utils.train_fn(
        reward_fn=None,
        output_dir=str(train_dir),
        run_name="smacv2_utils_test",
    )

    policy = _policy_from_experiment(experiment)
    env = getattr(experiment, "test_env", None)
    try:
        if env is None:
            env = smac_utils.create_env(task_name="terran_2_vs_4", num_envs=1, seed=seed, device="cpu")
        feature_names = _get_obs_feature_names(env)

        states, images, error = smac_utils.rollout_fn(
            policy=policy,
            env=env,
            max_steps=max_steps,
            seed=seed,
        )
        _print_raw_observations(
            states,
            label="smacv2.utils.train_fn + rollout_fn",
            feature_names=feature_names,
        )
        save_rollout(states, images or [], str(rollout_dir))

        return {
            "status": "ok" if error is None else "error",
            "error": None if error is None else str(error),
            "n_states": len(states or []),
            "n_images": len(images or []),
            "train_path": str(train_dir),
            "rollout_path": str(rollout_dir),
        }
    finally:
        _safe_close(env)


def _run_case(name: str, fn) -> dict[str, Any]:
    try:
        result = fn()
        result["case"] = name
        return result
    except Exception as exc:
        return {
            "case": name,
            "status": "exception",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SMACv2 rollout/train functions and save artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/smacv2_test"), help="Artifact output root")
    parser.add_argument("--max-steps", type=int, default=64, help="Rollout max steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--mode",
        choices=["all", "rollout", "train"],
        default="all",
        help="Run rollout-only, train-only, or both",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "mode": args.mode,
        "results": [],
    }

    if args.mode in {"all", "rollout"}:
        summary["results"].append(
            _run_case(
                "smacv2_common.rollout_smacv2_env",
                lambda: _run_common_rollout(
                    output_dir=output_dir / "smacv2_common" / "rollout",
                    max_steps=args.max_steps,
                    seed=args.seed,
                ),
            )
        )
        summary["results"].append(
            _run_case(
                "smacv2.utils.rollout_fn",
                lambda: _run_utils_rollout(
                    output_dir=output_dir / "smacv2_utils" / "rollout",
                    max_steps=args.max_steps,
                    seed=args.seed,
                ),
            )
        )

    if args.mode in {"all", "train"}:
        summary["results"].append(
            _run_case(
                "smacv2_common.train_smacv2_env + rollout_smacv2_env",
                lambda: _run_common_train_and_rollout(
                    output_dir=output_dir / "smacv2_common",
                    max_steps=args.max_steps,
                    seed=args.seed,
                ),
            )
        )
        summary["results"].append(
            _run_case(
                "smacv2.utils.train_fn + rollout_fn",
                lambda: _run_utils_train_and_rollout(
                    output_dir=output_dir / "smacv2_utils",
                    max_steps=args.max_steps,
                    seed=args.seed,
                ),
            )
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
