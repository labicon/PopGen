from typing import List, Any
import litellm
from .base import BehaviorGenerationStrategy, Behavior, BehaviorList

DIVIDER_GENERATION_PROMPT = \
"""You are analyzing a multi-agent environment to identify key strategic dimensions that differentiate player behaviors.

Environment Description:
{environment_description}

Existing strategic dividers:
{existing_dividers}

Identify a NEW strategic dimension (divider) that could meaningfully partition the space of possible player behaviors. This divider should be:
1. Orthogonal to existing dividers (if any)
2. Relevant to the environment
3. A spectrum or binary choice that players could position themselves on

Respond with just the divider description (e.g., "Risk tolerance: conservative vs aggressive" or "Cooperation level: selfish vs altruistic")."""

BEHAVIOR_GENERATION_PROMPT = \
"""You are designing diverse player behaviors for a multi-agent environment.

Environment Description:
{environment_description}

Strategic dividers to consider:
{dividers}

Generate a diverse set of player behaviors that span the strategic space defined by these dividers. Each behavior should represent a coherent strategy or personality. Aim to cover different combinations of the strategic dimensions.

Generate exactly 4 distinct behaviors."""


class StratifiedStrategy(BehaviorGenerationStrategy):
    def __init__(self, model: str = "openai/gpt-5"):
        super().__init__(model)

    def generate(self, environment_description: str, **kwargs: Any) -> List[Behavior]:
        """
        Generate behaviors by first identifying strategic dividers and then sampling from partitions.
        """
        # 1. Generate Divider 1
        divider1 = self._generate_single_divider(environment_description, [])

        # 2. Generate Divider 2
        divider2 = self._generate_single_divider(environment_description, [divider1])

        # 3. Generate Behaviors based on dividers
        behaviors = self._generate_behaviors_from_dividers(environment_description, divider1, divider2)

        return behaviors

    def _generate_single_divider(self, environment_description: str, existing_dividers: List[str]) -> str:
        prompt = DIVIDER_GENERATION_PROMPT.format(
            environment_description=environment_description,
            existing_dividers=self._list_to_string(existing_dividers)
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    def _generate_behaviors_from_dividers(self, environment_description: str, divider1: str, divider2: str) -> List[Behavior]:
        prompt = BEHAVIOR_GENERATION_PROMPT.format(
            environment_description=environment_description,
            dividers=self._list_to_string([divider1, divider2])
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=BehaviorList,
        )

        behavior_list = BehaviorList.model_validate_json(response.choices[0].message.content)
        self.behavior_list.extend(behavior_list.behaviors)

        return behavior_list.behaviors

    def _list_to_string(self, items: List[str]) -> str:
        if not items:
            return "None"
        return "\n".join(f"- {item}" for item in items)
