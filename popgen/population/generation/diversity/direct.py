import math
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
    """Generate diverse items by requesting a list directly from the LLM."""

    def __init__(
        self,
        model_config: LitellmConfig,
        output_type: type[T],
        output_dir: str | None = None,
        samples_per_turn: int | None = None,
        independent_multiturns: bool = False,
        independent_workers: int = 4,
    ):
        super().__init__(model_config, output_type, output_dir)
        self.samples_per_turn = samples_per_turn
        self.independent_multiturns = independent_multiturns
        self.independent_workers = independent_workers
        self._list_type = create_model(
            f"{output_type.__name__}List",
            items=(list[output_type], ...),
        )

    def generate(self, input_prompt: str, **kwargs: Any) -> list[T]:
        num_items = kwargs.get("num_items", 4)
        k = self.samples_per_turn or num_items

        # Single call — no batching needed
        if k >= num_items:
            items = self._generate_single(input_prompt=input_prompt, num_items=num_items)
            self.generated_items.extend(items)
            return items

        # Build batch sizes: [k, k, ..., remainder]
        n_batches = math.ceil(num_items / k)
        batch_sizes = [k] * (n_batches - 1) + [num_items - k * (n_batches - 1)]

        if self.independent_multiturns:
            from pqdm.threads import pqdm

            args = [
                {"input_prompt": input_prompt, "num_items": bs}
                for bs in batch_sizes
            ]
            results = pqdm(args, self._generate_single, n_jobs=min(n_batches, self.independent_workers), argument_type="kwargs", disable=True)
            items = [item for batch in results for item in batch]
        else:
            # Sequential: each batch sees previous generations as context
            items: list = []
            for bs in batch_sizes:
                if items:
                    prev_list = "\n".join(f"- {item.model_dump_json()}" for item in items)
                    batch_prompt = f"{input_prompt}\n\nGenerate {bs} alternatives to the ones already generated:\n{prev_list}"
                else:
                    batch_prompt = input_prompt

                batch_items = self._generate_single(input_prompt=batch_prompt, num_items=bs)
                items.extend(batch_items)

        self.generated_items.extend(items)
        return items

    def _generate_single(self, input_prompt: str, num_items: int) -> list[T]:
        """Generate items from a single LLM call."""
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
            return result.items[:num_items]
        except Exception as e:
            raise RuntimeError(f"Failed to generate items: {e}")
