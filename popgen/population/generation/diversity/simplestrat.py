from typing import List, Any, Type, TypeVar
from pydantic import BaseModel, create_model
import litellm

from .base import GenerationStrategy


T = TypeVar("T", bound=BaseModel)


DIVIDER_GENERATION_PROMPT = \
"""You are analyzing a multi-agent environment to identify key strategic dimensions that differentiate player {item_type}.

Environment Description:
{environment_description}

Item Description:
{item_description}

Existing strategic dividers:
{existing_dividers}

Identify a NEW strategic dimension (divider) that could meaningfully partition the space of possible player {item_type}. This divider should be:
1. Orthogonal to existing dividers (if any)
2. Relevant to the environment.

Respond with just the divider description (e.g. for player behavior, "Risk tolerance: conservative vs aggressive" or "Cooperation level: selfish vs altruistic")."""

GENERATION_PROMPT = \
"""You are designing diverse player {item_type} for a multi-agent environment.

Environment Description:
{environment_description}

Item Description:
{item_description}

Strategic dividers to consider:
{dividers}

Generate a diverse set of player {item_type} that span the strategic space defined by these dividers. 

Generate exactly {num_items} distinct {item_type}."""


class StratifiedStrategy(GenerationStrategy):
    """Generate diverse items by first identifying strategic dividers, then sampling from partitions."""

    def __init__(self, model: str, output_type: Type[T]):
        super().__init__(model, output_type)

    def generate(self, environment_description: str, **kwargs: Any) -> List[T]:
        """
        Generate items by first identifying strategic dividers and then sampling from partitions.
        """
        # 1. Generate Divider 1
        divider1 = self._generate_single_divider(environment_description, [])

        # 2. Generate Divider 2
        divider2 = self._generate_single_divider(environment_description, [divider1])

        # 3. Generate items based on dividers
        items = self._generate_items_from_dividers(environment_description, [divider1, divider2], num_items=kwargs.get("num_items", 4))

        return items

    def _generate_single_divider(self, environment_description: str, existing_dividers: List[str]) -> str:
        prompt = DIVIDER_GENERATION_PROMPT.format(
            environment_description=environment_description,
            existing_dividers=self._list_to_string(existing_dividers),
            item_type=self.output_type.__name__,
            item_description=self.output_type.__doc__
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    def _generate_items_from_dividers(self, environment_description: str, dividers: List[str], num_items: int) -> List[T]:
        prompt = GENERATION_PROMPT.format(
            environment_description=environment_description,
            dividers=self._list_to_string(dividers),
            item_type=self.output_type.__name__,
            item_description=self.output_type.__doc__,
            num_items=num_items,
        )

        # generate list response format dynamically based on output_type
        list_response_format = create_model(
            f"{self.output_type.__name__}List",
            items=(List[self.output_type], ...),
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=list_response_format,
        )

        item_list = list_response_format.model_validate_json(response.choices[0].message.content).items

        self.generated_items.extend(item_list)

        return item_list
