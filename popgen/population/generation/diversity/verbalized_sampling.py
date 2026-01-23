import json
import os
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, create_model

from popgen.utils.parsing import parse_llm_response

from .base import GenerationStrategy

T = TypeVar("T", bound=BaseModel)

VERBALIZED_SAMPLING_SYSTEM_PROMPT = """Generate {num_samplings} responses to the input prompt.
{target_words_instruction}
Return the responses in the following format:
    - item: {item_name}, {item_description}
    - probability: the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).
{cot_instruction}

{sampling_instruction}"""

COT_INSTRUCTION = """Additionally, provide a "reasoning" field explaining your thought process for generating diverse responses and estimating their probabilities."""


class VerbalizedSamplingStrategy(GenerationStrategy):
    """Generate diverse items by asking LLM to verbalize probability distributions."""

    def __init__(
        self,
        model: str,
        output_type: type[T],
        output_dir: str | None = None,
        probability_tuning: float | None = None,
        do_cot: bool = False,
        samples_per_turn: int | None = None,
    ):
        super().__init__(model, output_type, output_dir)
        self.probability_tuning = probability_tuning
        self.do_cot = do_cot
        self.samples_per_turn = samples_per_turn
        self._build_response_types()

    def _build_response_types(self) -> None:
        """Build dynamic response types based on output_type."""
        self.sampled_response_type = create_model(
            "SampledResponse",
            item=(self.output_type, ...),
            probability=(float, ...),
        )
        self.sampled_response_list_type = create_model(
            "SampledResponseList",
            responses=(list[self.sampled_response_type], ...),
        )
        self.sampled_response_list_type_cot = create_model(
            "SampledResponseListWithReasoning",
            reasoning=(str, ...),
            responses=(list[self.sampled_response_type], ...),
        )

    def generate(self, input_prompt: str, **kwargs: Any) -> list[T]:
        """
        Generate items by sampling from verbalized probability distributions.

        Args:
            input_prompt: A prompt describing the generation task.
            num_items: Number of items to generate.
            target_words: Optional target word count for each response.
        """
        num_items = kwargs.get("num_items", 4)
        target_words = kwargs.get("target_words", None)

        sampled_responses = self._generate_samples(
            input_prompt=input_prompt,
            num_samplings=num_items,
            target_words=target_words,
        )

        # Save samples with probabilities
        if self.output_dir:
            samples_data = [{"item": r.item.model_dump(), "probability": r.probability} for r in sampled_responses]
            samples_path = os.path.join(self.output_dir, "samples_with_probs.json")
            with open(samples_path, "w") as f:
                json.dump(samples_data, f, indent=2)

        items = [r.item for r in sampled_responses]

        self.generated_items.extend(items)

        return items

    def _generate_samples(
        self,
        input_prompt: str,
        num_samplings: int,
        target_words: int | None = None,
    ) -> list[T]:
        """Generate sampled responses with probabilities from the LLM."""
        if self.samples_per_turn is not None:
            return self._generate_samples_iterative(
                input_prompt=input_prompt,
                num_samplings=num_samplings,
                target_words=target_words,
            )

        return self._generate_samples_single(
            input_prompt=input_prompt,
            num_samplings=num_samplings,
            target_words=target_words,
        )

    def _generate_samples_iterative(
        self,
        input_prompt: str,
        num_samplings: int,
        target_words: int | None = None,
    ) -> list[T]:
        """Generate samples iteratively, providing past generations as context."""
        all_responses = []
        remaining = num_samplings

        while remaining > 0:
            batch_size = min(self.samples_per_turn, remaining)

            if all_responses:
                # Build list of past generations
                samples_list = "\n".join(f"- {r.item.model_dump_json()}" for r in all_responses)
                batch_prompt = f"{input_prompt}\n\nGenerate {batch_size} alternative responses to the ones already generated:\n{samples_list}"
            else:
                batch_prompt = input_prompt

            batch_responses = self._generate_samples_single(
                input_prompt=batch_prompt,
                num_samplings=batch_size,
                target_words=target_words,
            )
            all_responses.extend(batch_responses)
            remaining -= batch_size

        return all_responses

    def _generate_samples_single(
        self,
        input_prompt: str,
        num_samplings: int,
        target_words: int | None = None,
    ) -> list[T]:
        """Generate sampled responses with probabilities from a single LLM call."""
        if self.probability_tuning is not None:
            sampling_instruction = f"Randomly sample the responses from the distribution, with the probability of each response must be below {self.probability_tuning}."
        else:
            sampling_instruction = "Randomly sample the responses from the full distribution."

        target_words_instruction = f"Each response should be approximately {target_words} words.\n" if target_words is not None else ""

        system_prompt = VERBALIZED_SAMPLING_SYSTEM_PROMPT.format(
            num_samplings=num_samplings,
            target_words_instruction=target_words_instruction,
            sampling_instruction=sampling_instruction,
            cot_instruction=COT_INSTRUCTION if self.do_cot else "",
            item_name=self.output_type.__name__,
            item_description=self.output_type.__doc__,
        )

        response_type = self.sampled_response_list_type_cot if self.do_cot else self.sampled_response_list_type

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}],
                response_format=response_type,
                max_retries=3,
            )

            result = parse_llm_response(response, response_type)
            return result.responses

        except Exception as e:
            raise RuntimeError(f"Failed to generate samples: {e}")
