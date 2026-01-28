import litellm

from popgen.population.generation.base import GeneratorWithFeedback
from popgen.population.types import CodeSnippet
from popgen.utils.parsing import clean_code, parse_llm_response

SYSTEM_PROMPT = """You are an expert AI agent specializing in generating effective policies scripts for multi-agent coordination tasks.

Your task is to generate a policy script that aligns with the given behavior description given observations on the environment.

Some important details to consider:
- The policy should be provided as Python code that can be executed in a simulation environment.
- The policy should define a function `policy(obs)` that takes in the current observation of the ego agent and returns an action.
- The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.
- Only rely on `torch`, make sure to import it before function definition if necessary. Do not include any other imports.

The observation `obs` passed to the function is a torch tensor of shape (observation_dim,).
Exact information about the observation format can be found in observation function of the environment."""

POLICY_GENERATION_PROMPT = """Observation function of the environment:
{observation_format}

Expected output (action) format:
{action_format}

Behavior description:
{behavior}

Provide the python code that describes the behavior as the `policy(obs)` function, and a concise reasoning on the implementation."""


class PolicyCodeGenerator(GeneratorWithFeedback):
    """Generator for agent policies based on behavior descriptions."""

    def __init__(self, model: str):
        super().__init__(model, generation_type="PolicyCode")

    def generate(
        self,
        behavior: str,
        observation_format: str,
        action_format: str,
        feedbacks: list | None = None,
        **kwargs,
    ) -> CodeSnippet:
        """
        Generate a policy for the ego agent based on behavior description.

        Args:
            behavior: The behavior the policy should implement.
            observation_format: Description of the observation format.
            action_format: Description of the action format.
        Returns:
            Policy containing the generated code.
        """
        prompt = POLICY_GENERATION_PROMPT.format(observation_format=observation_format, action_format=action_format, behavior=behavior)

        response = super().generate(
            user_prompt=prompt,
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
