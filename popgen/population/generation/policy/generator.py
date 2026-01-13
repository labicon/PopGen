from typing import Optional
from pydantic import BaseModel
import litellm

from popgen.population.generation.generator import Generator

class Policy(BaseModel):
    """Generated policy code for an agent."""
    code: str
    agent_id: int

class PolicyCode(BaseModel):
    """Generated policy code for an agent."""
    code: str

POLICY_GENERATION_PROMPT = """You are an expert AI agent specializing in generating effective policies for multi-agent coordination tasks.

Your task is to generate a policy that aligns with the given behavior description.
The policy should be provided as Python code that can be executed in a simulation environment.
The policy should define a function `agent{agent_id}_policy(obs)` that takes in the current observation of the agent and returns an action.
The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.

The observation `obs` passed to the function is a torch tensor of shape (observation_dim,).
Exact information about the observation format can be found in observation function of the environment.
Observation function of the environment:
{observation_format}

Please provide ONLY the python code for the function `agent{agent_id}_policy`.
Do not include markdown formatting or explanations.
Only import torch before function definition if necessary. Do not include any other imports."""

PARTNER_POLICY_GENERATION_PROMPT = """You are an expert AI agent specializing in generating effective policies for multi-agent coordination tasks.
Your goal is to generate a policy for Agent {agent_id} that effectively coordinates with the partner agent.

You are provided with the partner agent's behavior description and its policy code.

Your task is to generate a policy for Agent {agent_id} (`agent{agent_id}_policy`) that coordinates with the partner's behavior to achieve the task.
The task is reaching a same landmark together in a 2D space.

The policy should be provided as Python code that can be executed in a simulation environment.
The policy should define a function `agent{agent_id}_policy(obs)` that takes in the current observation of the agent and returns an action.
The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.

The observation `obs` passed to the function is a torch tensor of shape (observation_dim,).
Exact information about the observation format can be found in observation function of the environment.
Observation function of the environment:
{observation_format}

Please provide ONLY the python code for the function `agent{agent_id}_policy`.
Do not include markdown formatting or explanations.
Only import torch before function definition if necessary. Do not include any other imports."""


class PolicyGenerator(Generator):
    """Generator for agent policies based on behavior descriptions."""

    def __init__(self, model: str):
        self.model = model

    def generate(
        self,
        behavior: str,
        observation_format: str,
        feedbacks: Optional[dict] = None,
    ) -> PolicyCode:
        """
        Generate a policy for agent0 based on behavior description.

        Args:
            behavior: The behavior the policy should implement.
            observation_format: Description of the observation format.
        
        Returns:
            Policy containing the generated code and agent_id.
        """
        # Generate agent0 policy
        agent0_policy = self._generate_standalone_policy(
            behavior=str(behavior),
            observation_format=observation_format,
            agent_id=0,
            feedbacks=feedbacks
        )

        # Generate agent1 policy that coordinates with agent0
        agent1_policy = self._generate_coordinating_policy(
            behavior=str(behavior),
            observation_format=observation_format,
            agent_id=1,
            partner_behavior=str(behavior),
            partner_policy=agent0_policy.code,
        )

        policy_code = agent0_policy.code + "\n\n" + agent1_policy.code

        return PolicyCode(code=policy_code)

    def _generate_standalone_policy(
        self,
        behavior: str,
        observation_format: str,
        agent_id: int,
        feedbacks: Optional[list] = None,
    ) -> Policy:
        """Generate a policy based solely on behavior description."""
        system_prompt = POLICY_GENERATION_PROMPT.format(
            agent_id=agent_id,
            observation_format=observation_format,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": behavior},
        ]

        if feedbacks is not None and len(feedbacks) > 0:
            feedback_previous = "\n\n".join(
                f" - Policy Code: {pol}\n   Feedback: {fdb}"
                for (pol, fdb) in feedbacks
            )
            refinement_message = (
                f"The previous attempts produced the following errors and feedbacks:\n"
                f"{feedback_previous}\n\n"
                f"Please fix the error and provide the corrected policy code."
            )
            messages.append({"role": "user", "content": refinement_message})

        response = litellm.completion(
            model=self.model,
            messages=messages,
        )

        code = self._clean_code(response.choices[0].message.content)
        return Policy(code=code, agent_id=agent_id)

    def _generate_coordinating_policy(
        self,
        behavior: str,
        observation_format: str,
        agent_id: int,
        partner_behavior: str,
        partner_policy: str,
    ) -> Policy:
        """Generate a policy that coordinates with a partner agent."""
        system_prompt = PARTNER_POLICY_GENERATION_PROMPT.format(
            agent_id=agent_id,
            observation_format=observation_format,
        )

        user_prompt = f"""Partner Agent Behavior:
{partner_behavior}

Partner Agent Policy Code:
```python
{partner_policy}
```

Your Agent Behavior:
{behavior}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = litellm.completion(
            model=self.model,
            messages=messages,
        )

        code = self._clean_code(response.choices[0].message.content)
        return Policy(code=code, agent_id=agent_id)

    def _clean_code(self, code: str) -> str:
        """Remove markdown code blocks from generated code."""
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
