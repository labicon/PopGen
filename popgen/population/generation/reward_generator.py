import litellm
from pydantic import BaseModel

from popgen.population.generation.base import BaseGenerator
from popgen.utils.parsing import clean_code, parse_llm_response


class Reward(BaseModel):
    """Generated reward function code for an agent."""

    code: str
    reasoning: str | None = None


REWARD_GENERATION_PROMPT = """You are an expert AI agent specializing in designing reward functions for reinforcement learning.

Your task is to generate a reward function for agent0 that incentivizes the given behavior.
The behavior will be given to you in JSON format with the following structure:
{{
    "name": str,
    "description": str
}}

The reward function should be provided as Python code that can be executed in a simulation environment.
The reward function should define a function `agent0_reward(obs, action, next_obs, done)` that computes a scalar reward.

Parameters:
- obs: The current observation (torch tensor)
- action: The action taken (torch tensor)
- next_obs: The next observation after taking the action (torch tensor)
- done: Whether the episode has ended (bool)

Returns:
- reward: A scalar float value

Codebase context (environment and observation format):
{codebase}

Provide the python code for the function `agent0_reward` and a concise reasoning on the implementation.
Only import torch before function definition if necessary. Do not include any other imports."""


class RewardFnGenerator(BaseGenerator):
    """Generator for agent reward functions based on behavior descriptions."""

    def __init__(self, model):
        self.model = model

    def generate(
        self,
        behavior: str,
        codebase_context: str,
        feedbacks: list | None = None,
    ) -> Reward:
        """
        Generate a reward function for agent0 based on behavior description.

        Args:
            behavior: The behavior the reward should incentivize.
            codebase_context: Context about the environment and observation format.
            previous_reward: Previous reward attempt (for refinement).
            error_message: Error from previous attempt (for refinement).

        Returns:
            Reward containing the generated code and optional reasoning.
        """
        system_prompt = REWARD_GENERATION_PROMPT.format(codebase=codebase_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": behavior},
        ]

        if feedbacks is not None and len(feedbacks) > 0:
            feedback_previous = "\n\n".join(f" - Reward Code: {rew}\n   Feedback: {fdb}" for (rew, fdb) in feedbacks)
            refinement_message = f"The previous attempts produced the following errors and feedbacks:\n{feedback_previous}\n\nPlease fix the errors and provide the corrected reward function."
            messages.append({"role": "user", "content": refinement_message})

        response = litellm.completion(
            model=self.model,
            messages=messages,
            response_format=Reward,
        )

        return parse_llm_response(
            response,
            Reward,
            postprocess_fn=lambda x: {"code": clean_code(x.code)},
        )
