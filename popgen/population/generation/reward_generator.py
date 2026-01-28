import litellm

from popgen.population.generation.base import GeneratorWithFeedback
from popgen.population.types import CodeSnippet
from popgen.utils.parsing import clean_code, parse_llm_response

SYSTEM_PROMPT = """You are an expert AI agent specializing in designing reward functions for reinforcement learning.

Your task is to generate a reward function for an ego agent that incentivizes the given behavior.

Your task is to generate a reward function that aligns with this behavior.
The function must be named `compute_reward(self, agent)` and return a reward for the given agent.
The reward should be a torch tensor of shape (batch_dim,) that compute the reward for each element of the batch.

The `agent` and `world` objects provide access to the simulation state.
Common attributes:
- `agent.state.pos`: Position of the agent (2D vector).
- `agent.state.vel`: Velocity of the agent (2D vector).
- `world.agents`: List of all agents (0-indexed).

Other attributes will be described in the environment description, observation format, or example reward function.

Please provide the python code for the function `compute_reward`, as well as a reasoning .
Torch is already imported. Do not include any other imports."""

REWARD_GENERATION_PROMPT = """The environment is described as follows:
{environment_description}

The observation format is as follows:
{observation_format}

The reward format is as follows:
{reward_format}

Behavior description:
{behavior}

Provide the reward function to train an agent with desired behavior as the `compute_reward(self, agent)` function, and a concise reasoning on the implementation."""


class RewardFnGenerator(GeneratorWithFeedback):
    """Generator for agent reward functions based on behavior descriptions."""

    def __init__(self, model):
        super().__init__(model, generation_type="RewardFn")

    def generate(
        self,
        behavior: str,
        environment_description: str,
        observation_format: str,
        reward_format: str,
        feedbacks: list | None = None,
        **kwargs,
    ) -> CodeSnippet:
        """
        Generate a reward function for agent0 based on behavior description.

        Args:
            behavior: The behavior the reward should incentivize.
            environment_description: Description of the environment.
            observation_format: Description of the observation format.
            reward_format: Description of the reward format.
            feedbacks: Optional feedbacks to refine generation.
        Returns:
            Reward containing the generated code and optional reasoning.
        """
        user_prompt = REWARD_GENERATION_PROMPT.format(environment_description=environment_description, observation_format=observation_format, reward_format=reward_format, behavior=behavior)

        response = super().generate(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            feedbacks=feedbacks,
            response_format=CodeSnippet,
            **kwargs,
        )

        return parse_llm_response(
            response,
            CodeSnippet,
            postprocess_fn=lambda x: {"code": clean_code(x.code)},
        )
