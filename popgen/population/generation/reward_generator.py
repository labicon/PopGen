from popgen.population.generation.base import GeneratorWithFeedback
from popgen.population.types import CodeSnippet
from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import clean_code, parse_llm_response

# ---------------------------------- PROMPTS --------------------------------- #

SYSTEM_PROMPT = """You are an expert AI agent specializing in designing reward functions for reinforcement learning in multi-agent coordination tasks.

Your task is to generate a reward function for an ego agent that incentivizes the given behavior.

Some important details to consider:
- The reward function must be named `compute_reward(self, agent)` and return a reward for the given agent.
- The exact return type/shape must follow the reward API reference below.
- `torch` is already imported. Do not include any other imports.
- Environment attributes are accessible via `self` and `agent` as described in the environment description or example reward functions.

Environment description:
{environment_description}

Reward API reference:
{reward_api_reference}

Output format requirements:
- Return a valid json object.
- The json object must contain exactly these keys: "code" and "reasoning".
- "code" must be a Python string containing only the reward function implementation.
- "reasoning" must be a detailed explanation of how the reward was designed and why it incentivizes the desired behavior.
- Do not wrap the json in markdown fences."""

REWARD_GENERATION_PROMPT = """# Behavior description:
{behavior}

Provide the python code that implements the reward as the `compute_reward(self, agent)` function and your reasoning.
Think step by step.

Return your final answer as a single valid json object with keys "code" and "reasoning"."""

# ---------------------------------------------------------------------------- #


class RewardFnGenerator(GeneratorWithFeedback):
    """Generator for agent reward functions based on behavior descriptions."""

    def __init__(self, model_config: LitellmConfig):
        super().__init__(model_config, generation_type="RewardFn")

    def generate(
        self,
        behavior: str,
        environment_description: str,
        reward_api_reference: str | None = None,
        examples: list | None = None,
        feedbacks: list | None = None,
        **kwargs,
    ) -> CodeSnippet:
        """
        Generate a reward function for agent0 based on behavior description.

        Args:
            behavior: The behavior the reward should incentivize.
            environment_description: Description of the environment.
            reward_api_reference: Explicit reference of accessible runtime attributes/methods.
            examples: Example reward functions to guide generation.
            feedbacks: Optional feedbacks to refine generation.
        Returns:
            Reward containing the generated code and optional reasoning.
        """
        system_prompt = SYSTEM_PROMPT.format(
            environment_description=environment_description,
            reward_api_reference=reward_api_reference or "Not provided.",
        )
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
