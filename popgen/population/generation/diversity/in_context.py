from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

from .base import GenerationStrategy

T = TypeVar("T", bound=BaseModel)

GENERATION_PROMPT = """You are designing diverse {item_name} for a multi-agent environment.

Environment Description:
{environment_description}

Item Description:
{item_description}

Previously generated {item_name}:
{previous_items}

Generate a NEW and UNIQUE {item_name} that is distinctly different from the previously generated ones.

Focus on creating diversity - the new {item_name} should explore a different aspect of the strategy space."""


class InContextStrategy(GenerationStrategy):
    """Generate diverse items using in-context learning with previously generated items."""

    def __init__(self, model: str, output_type: type[T]):
        super().__init__(model, output_type)

    def generate(self, environment_description: str, **kwargs: Any) -> list[T]:
        """
        Generate items in an in-context manner, using previously generated items as context.
        """
        items = []
        num_items = kwargs.get("num_items", 4)
        for _ in range(num_items):
            item = self._generate_single(environment_description, previous_items=items)
            items.append(item)
        return items

    def _generate_single(self, environment_description: str, previous_items: list[T] = []) -> T:
        """
        Generate a single item, taking into account previous items.
        """
        prompt = GENERATION_PROMPT.format(
            environment_description=environment_description,
            previous_items=self._list_to_string(previous_items),
            item_name=self.output_type.__name__,
            item_description=self.output_type.__doc__,
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=self.output_type,
                max_retries=3,
            )

            item = self.output_type.model_validate_json(response.choices[0].message.content)
            self.generated_items.append(item)

            return item
        except Exception as e:
            raise RuntimeError(f"Failed to generate item: {e}")
