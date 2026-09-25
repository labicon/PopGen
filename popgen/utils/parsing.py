import json
import re
from collections.abc import Callable

from pydantic import BaseModel, ValidationError


class LLMResponseParseError(ValueError):
    """Raised when an LLM response cannot be parsed into the expected schema."""


def _extract_text_content(content) -> str:
    """Normalize response message content into a string for JSON parsing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)
    return "" if content is None else str(content)


def parse_llm_response[T: BaseModel](
    response,
    model_class: type[T],
    postprocess_fn: Callable[[T], dict] | None = None,
) -> T:
    """
    Parse LLM response content into a Pydantic model with fallback strategies.

    Args:
        response: The LLM response object from litellm.completion.
        model_class: The Pydantic model class to parse into.
        postprocess_fn: Optional function that takes the parsed model and returns
            a dict of fields to update.

    Returns:
        An instance of the model_class.

    Raises:
        LLMResponseParseError: If the response cannot be parsed into the model after all fallback attempts.
    """
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    result = None
    if isinstance(parsed, model_class):
        result = parsed
    elif isinstance(parsed, dict):
        result = model_class.model_validate(parsed)

    content = _extract_text_content(getattr(message, "content", ""))

    # Strategy 1: Try direct JSON validation
    if result is None and content:
        try:
            result = model_class.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError, TypeError):
            pass

    # Strategy 2: Try to extract JSON from markdown code blocks
    if result is None and content:
        try:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                result = model_class.model_validate_json(json_match.group(1))
        except (ValidationError, json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: Try to find any JSON object in the content
    if result is None and content:
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = model_class.model_validate_json(json_match.group(0))
        except (ValidationError, json.JSONDecodeError, TypeError):
            pass

    # All strategies failed
    if result is None:
        raise LLMResponseParseError(f"Failed to parse LLM response into {model_class.__name__}. Response content:\n{content[:500]}...")

    # Apply postprocessing if provided
    if postprocess_fn is not None:
        updates = postprocess_fn(result)
        for key, value in updates.items():
            setattr(result, key, value)

    return result


# ------------------------------ PostProcess Fn ------------------------------ #


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
