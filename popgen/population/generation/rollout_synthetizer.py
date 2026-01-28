"""Rollout synthetizer."""

from __future__ import annotations

from typing import Any

import litellm
import numpy as np
from pydantic import BaseModel

from popgen.utils.logging import encode_images_to_base64, rollout_states_to_str

# ---------------------------------- PROMPTS --------------------------------- #

BEHAVIOR_ANALYSIS_SYSTEM_PROMPT = """You are an expert in analyzing agent behavior.
You will be provided with rollouts from an agent (trajectory states and possibly images sequences).
Your task is to analyze the behavior of the ego agent based on these input rollouts.
Focus on the ego agent only.
Describe the observed behavior, giving it a name and a concise, key-point focused description.
Avoid verbosity. Focus on the core actions and strategies.
Provide concise reasoning for your analysis."""

BEHAVIOR_ANALYSIS_USER_PROMPT = """Environment Description:
{env_description}

Ego Agent Name: {ego_name}
Ego Agent Visual Description: {ego_visual_description}

Please analyze the behavior shown in the following rollout #{rollout_index}.
"""

SUMMARY_ANALYSIS_SYSTEM_PROMPT = "You are an expert in summarizing behavioral observations into concise, key-point focused descriptions."

SUMMARY_ANALYSIS_USER_PROMPT = """Environment Description:
{env_description}

Here are the behavior analyses from {n_analyses} different rollouts of the same agent.
Please summarize these observations into a single, concise behavior description that captures the general policy of the agent.
Focus on the most important key points. Avoid unnecessary details.

{analyses_text}

Provide a unified name, concise description, and short reasoning that explains the consistent behavior observed across rollouts."""

# ---------------------------------------------------------------------------- #


class RolloutSynthetizerResult(BaseModel):
    """Result of synthesizing rollouts."""

    name: str  # Name of the behavior
    description: str  # Description of the behavior
    reasoning: str  # Reasoning for the analysis

    def __str__(self) -> str:
        return f"Behavior: {self.name}\nDescription: {self.description}\nReasoning: {self.reasoning}"


class RolloutSynthetizer:
    """Synthetizer for agent behavior based on rollout trajectories."""

    def __init__(self, model: str):
        self.model = model

    def analyze_rollouts(
        self,
        ego_name: str,
        ego_visual_description: str,
        env_description: str,
        rollouts_states: list[dict[str, Any]],
        rollouts_images: list[list[np.ndarray]] = None,
        **kwargs,
    ) -> RolloutSynthetizerResult:
        """Synthesize behavior based on multiple rollouts.

        Args:
            ego_name: Name of the ego agent.
            ego_visual_description: Visual description of the ego agent.
            env_description: Description of the environment.
            rollouts_states: List of dictionary of rollout states.
            rollouts_images: List of list of rollout images.
        """
        if rollouts_images is None:
            rollouts_images = [[] for _ in rollouts_states]

        # 1. Analyze each rollout individually
        individual_results = []
        for i, (states, images) in enumerate(zip(rollouts_states, rollouts_images)):
            result = self._analyze_single_rollout(
                ego_name=ego_name,
                ego_visual_description=ego_visual_description,
                env_description=env_description,
                rollout_states=states,
                rollout_images=images,
                rollout_index=i,
                **kwargs,
            )
            individual_results.append(result)

        # 2. Summarize the results
        return self._summarize_analyses(individual_results, env_description=env_description, **kwargs)

    def _analyze_single_rollout(
        self,
        ego_name: str,
        ego_visual_description: str,
        env_description: str,
        rollout_states: dict[str, Any],
        rollout_images: list[np.ndarray],
        rollout_index: int,
        **kwargs,
    ) -> RolloutSynthetizerResult:
        """Analyze behavior for a single rollout.

        Args:
            ego_name: Name of the ego agent.
            ego_visual_description: Visual description of the ego agent.
            env_description: Description of the environment.
            rollout_states: Dictionary of rollout states.
            rollout_images: List of rollout images (numpy arrays).
            rollout_index: Index of the rollout.

        Returns:
            RolloutSynthetizerResult with name, description, and reasoning.
        """
        user_prompt = BEHAVIOR_ANALYSIS_USER_PROMPT.format(
            ego_name=ego_name,
            ego_visual_description=ego_visual_description,
            env_description=env_description,
            rollout_index=rollout_index + 1,
        )
        content = [{"type": "text", "text": user_prompt}]

        prefix = f"Rollout (showing {len(rollout_states)} sampled states and {len(rollout_images)} sampled images):"
        separator = "-" * 60
        content.append({"type": "text", "text": f"{prefix}\n{separator}"})

        # Add all images
        for encoded_string in encode_images_to_base64(rollout_images):
            # Optimistically truncate huge images if needed (handled by encode_images_to_base64 usually returning manageable strings or externally)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                }
            )

        # Add all states
        content.append({"type": "text", "text": rollout_states_to_str(rollout_states)})
        content.append({"type": "text", "text": separator})

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": BEHAVIOR_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=RolloutSynthetizerResult,
                **kwargs,
            )
            result = RolloutSynthetizerResult.model_validate_json(response.choices[0].message.content)
            return result
        except Exception as e:
            return RolloutSynthetizerResult(name="Error", description="Analysis failed.", reasoning=f"[ERROR] Analysis failed: {e}")

    def _summarize_analyses(
        self,
        analyses: list[RolloutSynthetizerResult],
        env_description: str,
        **kwargs,
    ) -> RolloutSynthetizerResult:
        """Summarize multiple behavior analyses into a single result."""

        analyses_text = "\n".join([f"Rollout {i + 1}:\nName: {r.name}\nDescription: {r.description}\nReasoning: {r.reasoning}\n" for i, r in enumerate(analyses)])

        summary_prompt = SUMMARY_ANALYSIS_USER_PROMPT.format(n_analyses=len(analyses), analyses_text=analyses_text, env_description=env_description)

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARY_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
                response_format=RolloutSynthetizerResult,
                **kwargs,
            )
            return RolloutSynthetizerResult.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            return RolloutSynthetizerResult(
                name="Error",
                description="Summary failed.",
                reasoning=f"[ERROR] Summary failed: {e}\n\nIndividual Analyses:\n{analyses_text}",
            )
