from popgen.population.generation.base import GeneratorWithFeedback
from popgen.population.types import CodeSnippet
from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import clean_code, parse_llm_response

# ---------------------------------- PROMPTS --------------------------------- #

SYSTEM_PROMPT = """You are an expert AI agent specializing in generating effective policies scripts for multi-agent coordination tasks.

Your task is to generate a policy script that aligns with the given behavior description given observations on the environment.

Some important details to consider:
- The policy should be provided as Python code that can be executed in a simulation environment.
- The policy should define a function `policy(obs)` that takes in the current observation of the ego agent and returns an action as a torch tensor.
- Only rely on `torch`, make sure to import it before function definition if necessary. Do not include any other imports.

The observation `obs` passed to the function is a torch tensor of shape (observation_dim,).

Environment description:
{environment_description}

Please ensure that the generated code is clean, well-structured, and adheres to the specified action format.
The returned action should be a torch tensor of shape (action_dim,) where action_dim is defined in the action format description.
In example for a continuous action space with 3 dimensions, the returned action should be something like `torch.tensor([0.5, -0.2, 1.0])`.

Output format requirements:
- Return a valid json object.
- The json object must contain exactly these keys: "code" and "reasoning".
- "code" must be a Python string containing only the policy implementation.
- "reasoning" must be the detailed explanation of how the policy was generated and why it is appropriate for the given behavior.
- Do not wrap the json in markdown fences."""

POLICY_GENERATION_PROMPT = """# Behavior description:
{behavior}

Provide the python code that describes the behavior as the `policy(obs)` function and your reasoning to arrive.
Think step by step.

Return your final answer as a single valid json object with keys "code" and "reasoning"."""

# ---------------------------------------------------------------------------- #


class PolicyCodeGenerator(GeneratorWithFeedback):
    """Generator for agent policies based on behavior descriptions."""

    def __init__(self, model_config: LitellmConfig):
        super().__init__(model_config, generation_type="PolicyCode")

    def generate(
        self,
        behavior: str,
        environment_description: str,
        examples: list | None = None,
        feedbacks: list | None = None,
        **kwargs,
    ) -> CodeSnippet:
        """
        Generate a policy for the ego agent based on behavior description.

        Args:
            behavior: The behavior the policy should implement.
            environment_description: Description of the environment.
            examples: Example policies to guide generation.
        Returns:
            Policy containing the generated code.
        """
        system_prompt = SYSTEM_PROMPT.format(environment_description=environment_description)
        prompt = POLICY_GENERATION_PROMPT.format(behavior=behavior)

        response = super().generate(
            user_prompt=prompt,
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
