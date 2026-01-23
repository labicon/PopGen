import json
import math
import os
import random
from typing import Any, TypeVar

import litellm
import numpy as np
from pqdm.threads import pqdm
from pydantic import BaseModel, create_model

from popgen.utils.parsing import parse_llm_response

from .base import GenerationStrategy

T = TypeVar("T", bound=BaseModel)

STRATIFICATION_SYSTEM_PROMPT = """You are analyzing key strategic dimensions (properties) that most differentiate several {item_type}.
Your task is to return a list of True/False properties that can be used to partition the space of possible {item_type}."""

STRATIFICATION_USER_PROMPT = """# Your Task
Help me brainstorm how to respond to the user request by providing a list of True/False properties the solution may or may not have.
Use the following step-by-step to come up with good properties:
    1. If you were playing 20 questions, what's a good first question to ask that would split the possibilities in half?
    List at least {num_candidates} questions and their corresponding properties.

    Question: <Description>

    2. Rewrite each question as a True/False property that's true for one half and false for the other.

    Question: <Description>
    True/False Property: <Property Description>

    3. For each property, come up with an example that would satisfy the property.

    Property: <Description>
    Example: <Description>

    Is it a valid answer to the user's request? <Yes/No>

    4. For each property, come up with an example that would not satisfy the property.

    Property: <Description>
    Example: <Description>

    Is it a valid answer to the user's request? <Yes/No>

    5. Does the property mention a candidate answer in it?

    Property: <Description>
    Does the property mention a candidate answer in it? <Yes/No>

    6. For each property, list whether we should include it or not in the final list of properties.
    Do not include ones where an example from above is not valid or if it mentions a candidate answer in it.

    Property: <Description>
    Include in final list? <Yes/No>

Ensure all properties are listed are sentences that are either True or False.

# User Request
```
{user_request}
```

# Answer Format
Return the final list of properties as a list of descriptions, i.e.:
```
- <Property Description 1>
[...]
- <Property Description N>
```

The final list should contain at least {num_properties} properties!
"""

GENERATION_SYSTEM_PROMPT = "You are a helpful assistant that generates {item_type} based on specified properties."
GENERATION_PROMPT = """Input:
{input_prompt}

Properties to respect generating the {item_type}:
{properties}

Generate a single {item_type} that respects these properties."""


class PropertyList(BaseModel):
    properties: list[str]


class StratifiedStrategy(GenerationStrategy):
    """Generate diverse items by first identifying strategic properties, then sampling from partitions."""

    def __init__(self, model: str, output_type: type[T], output_dir: str | None = None, candidate_upsampling: float = 1.0, num_workers: int = 1):
        super().__init__(model, output_type, output_dir)
        self.candidate_upsampling = candidate_upsampling
        self.num_workers = num_workers

    def generate(self, input_prompt: str, **kwargs: Any) -> list[T]:
        """
        Generate items by first identifying strategic properties and then sampling from partitions.
        """
        num_items = kwargs.get("num_items", 4)
        num_properties = kwargs.get("num_properties", math.ceil(np.log2(num_items)) + 1)

        # Generate properties
        properties = self.generate_properties(input_prompt, num_properties=num_properties, candidate_rate=self.candidate_upsampling)

        # Save properties
        if self.output_dir:
            properties_path = os.path.join(self.output_dir, "properties.json")
            with open(properties_path, "w") as f:
                json.dump(properties, f, indent=2)

        # Generate items based on properties
        # Pre-sample property sets for each item
        sampled_property_sets = [[prop for prop in properties if random.random() < 0.5] for _ in range(num_items)]

        def generate_single_item(stratum_set: list[str]) -> dict:
            item = self.generate_item_from_properties(input_prompt, stratum_set)
            return {
                "item": item,
                "sampled_properties": stratum_set,
            }

        results = pqdm(sampled_property_sets, generate_single_item, n_jobs=self.num_workers, desc="Generating items", disable=True)

        items_with_properties = []
        for result in results:
            self.generated_items.append(result["item"])
            items_with_properties.append(
                {
                    "item": result["item"].model_dump(),
                    "sampled_properties": result["sampled_properties"],
                }
            )

        # Save items with their sampled properties
        if self.output_dir:
            items_path = os.path.join(self.output_dir, "items_with_properties.json")
            with open(items_path, "w") as f:
                json.dump(items_with_properties, f, indent=2)

        return self.generated_items

    def generate_properties(self, input_prompt: str, num_properties: int, candidate_rate: float) -> list[str]:
        """
        Generate a list of strategic properties.
        """
        stratification_prompt = STRATIFICATION_USER_PROMPT.format(
            user_request=input_prompt,
            num_candidates=int(num_properties * candidate_rate),
            num_properties=num_properties,
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "system", "content": STRATIFICATION_SYSTEM_PROMPT.format(item_type=self.output_type.__name__)}, {"role": "user", "content": stratification_prompt}],
            response_format=PropertyList,
        )

        properties = parse_llm_response(response, PropertyList).properties

        assert len(properties) > 0, "No properties generated."

        return properties

    def generate_item_from_properties(self, input_prompt: str, properties: list[str]) -> T:
        """
        Generate items based on the given strategic properties.
        """
        system_prompt = GENERATION_SYSTEM_PROMPT.format(item_type=self.output_type.__name__)
        generation_prompt = GENERATION_PROMPT.format(
            input_prompt=input_prompt,
            properties=self._list_to_string(properties),
            item_type=self.output_type.__name__,
            item_description=self.output_type.__doc__,
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": generation_prompt}],
            response_format=self.output_type,
        )

        item = parse_llm_response(response, self.output_type)

        return item
