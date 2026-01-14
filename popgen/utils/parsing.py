from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_llm_response(
    response,
    model_class: type[T],
    postprocess_fn: Callable[[T], dict] | None = None,
) -> T:
    """
    Parse LLM response content into a Pydantic model.

    Args:
        response: The LLM response object from litellm.completion.
        model_class: The Pydantic model class to parse into.
        postprocess_fn: Optional function that takes the parsed model and returns
            a dict of fields to update.

    Returns:
        An instance of the model_class.
    """
    result = model_class.model_validate_json(response.choices[0].message.content)
    if postprocess_fn is not None:
        updates = postprocess_fn(result)
        for key, value in updates.items():
            setattr(result, key, value)
    return result


def clean_code(code: str) -> str:
    """Remove markdown code blocks from generated code."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()
