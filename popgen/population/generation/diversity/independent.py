from typing import Any, TypeVar

import litellm
from pqdm.threads import pqdm
from pydantic import BaseModel

from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import parse_llm_response

from .base import GenerationStrategy

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = """Generate a single {item_name} for the given input prompt.
{item_description}"""


class IndependentStrategy(GenerationStrategy):
    """Generate diverse items via independent parallel LLM calls (one call per item)."""

    def __init__(self, model_config: LitellmConfig, output_type: type[T], output_dir: str | None = None, num_workers: int = 4):
        super().__init__(model_config, output_type, output_dir)
        self.num_workers = num_workers

    def generate(self, input_prompt: str, **kwargs: Any) -> list[T]:
        num_items = kwargs.get("num_items", 4)

        system_prompt = SYSTEM_PROMPT.format(
            item_name=self.output_type.__name__,
            item_description=self.output_type.__doc__ or "",
        )

        def generate_single(_: int) -> T:
            response = litellm.completion(
                model=self.model_config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
                ],
                response_format=self.output_type,
                max_retries=3,
                **self.model_config.get_completion_kwargs(),
            )
            return parse_llm_response(response, self.output_type)

        items = pqdm(range(num_items), generate_single, n_jobs=self.num_workers, desc="Generating items", disable=True)

        self.generated_items.extend(items)
        return items
