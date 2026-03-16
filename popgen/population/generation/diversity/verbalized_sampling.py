import json
import math
import os
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, create_model

from popgen.utils.config import LitellmConfig
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

RERANKER_SYSTEM_PROMPT = """You are a behavior curator. You are given {num_candidates} candidate behaviors generated for the task below.

Produce a final curated list of exactly {num_items} behaviors that jointly maximize:
1. DIVERSITY — each behavior must explore a clearly distinct strategy, playstyle, or approach; no two should be variations of the same idea.
2. QUALITY — each behavior must be realistic (feasible in the game environment) and specific (concrete enough to implement directly, not vague).

For each output behavior you may:
- Copy a candidate verbatim if it already meets both criteria.
- Refine a candidate to make it more specific, realistic, or clearly distinct from others.
- Synthesize a new behavior by combining complementary aspects of two or more candidates, as long as the result is coherent.

First, write your reasoning: survey the candidates, identify clusters of similar ideas, note quality gaps, and decide your curation plan. Then output exactly {num_items} behaviors."""


class VerbalizedSamplingStrategy(GenerationStrategy):
    """Generate diverse items by asking LLM to verbalize probability distributions."""

    def __init__(
        self,
        model_config: LitellmConfig,
        output_type: type[T],
        output_dir: str | None = None,
        probability_tuning: float | None = None,
        do_cot: bool = False,
        samples_per_turn: int | None = None,
        oversampling_ratio: float = 1,
        independent_multiturns: bool = False,
        independent_workers: int = 4,
    ):
        super().__init__(model_config, output_type, output_dir)
        self.probability_tuning = probability_tuning
        self.do_cot = do_cot
        self.samples_per_turn = samples_per_turn
        self.oversampling_ratio = oversampling_ratio
        self.independent_multiturns = independent_multiturns
        self.independent_workers = independent_workers
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
        self.reranker_response_type = create_model(
            "RerankerResponse",
            reasoning=(str, ...),
            behaviors=(list[self.output_type], ...),
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
            num_items=num_items,
            target_words=target_words,
        )

        # Save all candidates (with probabilities) before any reranking
        if self.output_dir:
            samples_data = [{"item": r.item.model_dump(), "probability": r.probability} for r in sampled_responses]
            samples_path = os.path.join(self.output_dir, "samples_with_probs.json")
            with open(samples_path, "w") as f:
                json.dump(samples_data, f, indent=2)

        # Rerank: curate/compose num_items from the larger candidate pool
        if self.oversampling_ratio > 1 and len(sampled_responses) > num_items:
            items = self._rerank_samples(
                input_prompt=input_prompt,
                candidates=sampled_responses,
                num_items=num_items,
            )
        else:
            items = [r.item for r in sampled_responses]

        self.generated_items.extend(items)

        return items

    def _generate_samples(
        self,
        input_prompt: str,
        num_items: int,
        target_words: int | None = None,
    ) -> list[T]:
        """Generate sampled responses with probabilities from the LLM.

        ``samples_per_turn`` controls batching: the total request is split into
        ceil(total / k) calls of k items each.

        When ``independent_multiturns`` is True, batches run in parallel with
        no shared context.  Otherwise batches run sequentially and each call
        sees previously generated items as context.
        """
        ratio = max(1, int(self.oversampling_ratio))
        total_items = num_items * ratio
        k = self.samples_per_turn or total_items

        # Single call — no batching needed
        if k >= total_items:
            return self._generate_samples_single(
                input_prompt=input_prompt,
                num_samplings=total_items,
                target_words=target_words,
            )

        # Build batch sizes: [k, k, ..., remainder]
        n_batches = math.ceil(total_items / k)
        batch_sizes = [k] * (n_batches - 1) + [total_items - k * (n_batches - 1)]

        if self.independent_multiturns:
            from pqdm.threads import pqdm

            args = [{"input_prompt": input_prompt, "num_samplings": bs, "target_words": target_words} for bs in batch_sizes]
            results = pqdm(args, self._generate_samples_single, n_jobs=min(n_batches, self.independent_workers), argument_type="kwargs", disable=True)
            return [r for batch in results for r in batch]

        # Sequential: each batch sees previous generations as context
        all_responses: list = []
        for bs in batch_sizes:
            if all_responses:
                samples_list = "\n".join(f"- {r.item.model_dump_json()}" for r in all_responses)
                batch_prompt = f"{input_prompt}\n\nGenerate {bs} alternative responses to the ones already generated:\n{samples_list}"
            else:
                batch_prompt = input_prompt

            batch_responses = self._generate_samples_single(
                input_prompt=batch_prompt,
                num_samplings=bs,
                target_words=target_words,
            )
            all_responses.extend(batch_responses)

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
                model=self.model_config.model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}],
                response_format=response_type,
                max_retries=3,
                **self.model_config.get_completion_kwargs(),
            )

            result = parse_llm_response(response, response_type)
            return result.responses

        except Exception as e:
            raise RuntimeError(f"Failed to generate samples: {e}")

    def _rerank_samples(
        self,
        input_prompt: str,
        candidates: list,
        num_items: int,
    ) -> list[T]:
        """Curate num_items final behaviors from candidates.

        The LLM may select, refine, or compose behaviors from the candidate pool
        to maximize diversity and quality (realism + specificity).
        Returns a list of output_type instances directly.
        """
        num_candidates = len(candidates)
        system_prompt = RERANKER_SYSTEM_PROMPT.format(
            num_candidates=num_candidates,
            num_items=num_items,
        )

        candidates_text = "\n".join(f"[{i}] {c.item.model_dump_json()}" for i, c in enumerate(candidates))
        user_message = f"Task:\n{input_prompt}\n\nCandidates:\n{candidates_text}"

        try:
            response = litellm.completion(
                model=self.model_config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format=self.reranker_response_type,
                max_retries=3,
                **self.model_config.get_completion_kwargs(),
            )

            result = parse_llm_response(response, self.reranker_response_type)
            behaviors = result.behaviors[:num_items]

            # Fall back to original candidates if reranker returned too few
            if len(behaviors) < num_items:
                fallback = [c.item for c in candidates]
                behaviors.extend(fallback[: num_items - len(behaviors)])

            return behaviors

        except Exception as e:
            raise RuntimeError(f"Failed to rerank samples: {e}")
