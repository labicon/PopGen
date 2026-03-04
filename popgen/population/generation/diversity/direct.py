from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, create_model

from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import parse_llm_response

from .base import GenerationStrategy

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = """Generate exactly {num_items} diverse {item_name} for the given input prompt.
{item_description}
Make each {item_name} distinctly different from the others — maximize variety across the full space of possibilities."""


class DirectStrategy(GenerationStrategy):
    """Generate diverse items in a single LLM call by requesting a list directly."""

    def __init__(self, model_config: LitellmConfig, output_type: type[T], output_dir: str | None = None):
        super().__init__(model_config, output_type, output_dir)
        self._list_type = create_model(
            f"{output_type.__name__}List",
            items=(list[output_type], ...),
        )

    def generate(self, input_prompt: str, **kwargs: Any) -> list[T]:
        num_items = kwargs.get("num_items", 4)

        system_prompt = SYSTEM_PROMPT.format(
            num_items=num_items,
            item_name=self.output_type.__name__,
            item_description=self.output_type.__doc__ or "",
        )

        try:
            response = litellm.completion(
                model=self.model_config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
                ],
                response_format=self._list_type,
                max_retries=3,
                **self.model_config.get_completion_kwargs(),
            )
            result = parse_llm_response(response, self._list_type)
            items = result.items[:num_items]
        except Exception as e:
            raise RuntimeError(f"Failed to generate items: {e}")

        self.generated_items.extend(items)
        return items
