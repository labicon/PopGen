"""Cooperator training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from popgen.agents.callbacks import TeamRewardSumCallback
from popgen.cooperator.util import load_policies, select_policy_refs_from_behavior_results
from popgen.utils.run_names import (
    build_cooperator_run_name,
    extract_run_date_from_path,
    infer_env_name,
)


class CooperatorTrainingPipeline:
    """Trains a cooperator agent against a pre-generated teammate population."""

    def __init__(
        self,
        *,
        env_module: Any,
        results_dir: str | Path,
        output_dir: str | Path,
        run_name: str | None = None,
        reward_fn=None,
        train_kwargs: dict | None = None,
        sum_team_reward: bool = True,
    ):
        self.env_module = env_module
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.reward_fn = reward_fn
        self.train_kwargs = dict(train_kwargs or {})
        self.sum_team_reward = sum_team_reward

    def train_cooperator(self) -> Any:
        """Train cooperator against teammate policies loaded from results_dir."""
        refs = select_policy_refs_from_behavior_results(self.results_dir)
        if not refs:
            raise ValueError("No loadable teammate policy artifacts found in results_dir.")
        policies = load_policies(refs, self.env_module)
        if self.run_name:
            run_name = self.run_name
        else:
            env_name = infer_env_name(self.env_module)
            run_date = extract_run_date_from_path(self.output_dir)
            run_name = build_cooperator_run_name(env_name=env_name, run_date=run_date)
        train_kwargs = dict(self.train_kwargs)
        train_kwargs.pop("run_name", None)

        if self.sum_team_reward:
            extra = list(train_kwargs.get("callbacks", None) or [])
            extra.append(TeamRewardSumCallback())
            train_kwargs["callbacks"] = extra

        return self.env_module.train_fn(
            reward_fn=self.reward_fn,
            output_dir=str(self.output_dir),
            run_name=run_name,
            teammate_policy=[policies],
            **train_kwargs,
        )
