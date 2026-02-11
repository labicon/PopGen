from popgen.population.generation.base import GeneratorWithFeedback
from popgen.population.types import CodeSnippet
from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import clean_code, parse_llm_response

# ---------------------------------- PROMPTS --------------------------------- #

SYSTEM_PROMPT = """You are an expert AI agent specializing in designing reward functions for reinforcement learning.

Your task is to generate a reward function for an ego agent that incentivizes the given behavior.
The function must be named `compute_reward(self, agent)` and return a reward for the given agent.
The reward should be a torch tensor of shape (batch_dim,) that compute the reward for each element of the batch.
Environment attributes will be described in the environment description or example reward function.

# Environment description:
{environment_description}

Please provide the python code for the function `compute_reward`, as well as a reasoning.
Torch is already imported. Do not include any other imports."""

REWARD_GENERATION_PROMPT = """Behavior description:
{behavior}

Provide the reward function to train an agent with desired behavior as the `compute_reward(self, agent)` function, and a concise reasoning on the implementation."""

# ---------------------------------------------------------------------------- #


class RewardFnGenerator(GeneratorWithFeedback):
    """Generator for agent reward functions based on behavior descriptions."""

    def __init__(self, model_config: LitellmConfig):
        super().__init__(model_config, generation_type="RewardFn")

    def generate(
        self,
        behavior: str,
        environment_description: str,
        examples: list | None = None,
        feedbacks: list | None = None,
        **kwargs,
    ) -> CodeSnippet:
        """
        Generate a reward function for agent0 based on behavior description.

        Args:
            behavior: The behavior the reward should incentivize.
            environment_description: Description of the environment.
            examples: Example reward functions to guide generation.
            feedbacks: Optional feedbacks to refine generation.
        Returns:
            Reward containing the generated code and optional reasoning.
        """
        system_prompt = SYSTEM_PROMPT.format(environment_description=environment_description)
        user_prompt = REWARD_GENERATION_PROMPT.format(behavior=behavior)

        response = super().generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            examples=examples,
            feedbacks=feedbacks,
            response_format=CodeSnippet,
            **kwargs,
        )

        return parse_llm_response(
            response,
            CodeSnippet,
            postprocess_fn=lambda x: {"code": clean_code(x.code)},
        )
