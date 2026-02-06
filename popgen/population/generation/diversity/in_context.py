from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import parse_llm_response

from .base import GenerationStrategy

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = "You are a helpful assistant that diverse {item_name} based on an input prompt and previously generated items."
GENERATION_PROMPT = """Input:
{input_prompt}

Previously generated {item_name}:
{previous_items}

Generate a NEW and UNIQUE {item_name} that is distinctly different from the previously generated ones.
Focus on creating diversity - the new {item_name} should explore a different aspect of the space of possibilities."""


class InContextStrategy(GenerationStrategy):
    """Generate diverse items using in-context learning with previously generated items."""

    def __init__(self, model_config: LitellmConfig, output_type: type[T], output_dir: str | None = None):
        super().__init__(model_config, output_type, output_dir)

    def generate(self, input_prompt: str, **kwargs: Any) -> list[T]:
        """
        Generate items in an in-context manner, using previously generated items as context.
        """
        items = []
        num_items = kwargs.get("num_items", 4)
        for _ in range(num_items):
            item = self._generate_single(input_prompt, previous_items=items)
            self.generated_items.append(item)
            items.append(item)
        return items

    def _generate_single(self, input_prompt: str, previous_items: list[T] = []) -> T:
        """
        Generate a single item, taking into account previous items.
        """
        system_prompt = SYSTEM_PROMPT.format(item_name=self.output_type.__name__)
        generation_prompt = GENERATION_PROMPT.format(
            previous_items=self._list_to_string(previous_items),
            item_name=self.output_type.__name__,
            item_description=self.output_type.__doc__,
            input_prompt=input_prompt,
        )

        try:
            response = litellm.completion(
                model=self.model_config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": generation_prompt},
                ],
                response_format=self.output_type,
                max_retries=3,
                **self.model_config.get_completion_kwargs(),
            )

            item = parse_llm_response(response, self.output_type)

            return item
        except Exception as e:
            raise RuntimeError(f"Failed to generate item: {e}")
