"""Configuration utilities for LLM generation."""

import logging
from typing import Any

from pydantic import BaseModel, Field


class LitellmConfig(BaseModel):
    """Configuration for LiteLLM API calls.

    This config encapsulates all parameters needed for LiteLLM generation,
    including model selection, API credentials, and generation parameters.

    Attributes:
        model_name: The model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        api_key: Optional API key for the model provider.
        api_base: Optional custom API base URL (e.g., for local vLLM servers).
        generation_kwargs: Additional keyword arguments passed to litellm.completion().
            Common examples: temperature, top_p, max_tokens, response_format, etc.

    Example:
        >>> from popgen.utils.config import LitellmConfig
        >>>
        >>> # Simple config
        >>> config = LitellmConfig(model_name="openai/gpt-4o-mini")
        >>>
        >>> # With custom API base (e.g., local vLLM)
        >>> config = LitellmConfig(
        ...     model_name="hosted_vllm/Qwen/Qwen3-4B-Instruct",
        ...     api_base="http://localhost:8000/v1",
        ...     generation_kwargs={"temperature": 0.7, "max_tokens": 2000}
        ... )
        >>>
        >>> # Use in generator
        >>> from popgen.population.generation import TrajectoryGenerator
        >>> generator = TrajectoryGenerator(config=config)
    """

    model_name: str = Field(description="Model identifier for LiteLLM (e.g., 'openai/gpt-4')")
    api_base: str | None = Field(default=None, description="Custom API base URL")
    generation_kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional kwargs for litellm.completion() (temperature, top_p, etc.)")

    def __post_init__(self):
        """Post-initialization to validate generation_kwargs."""
        if "hosted_vllm" in self.model_name and self.api_base is None:
            self.api_base = "http://localhost:8000/v1"
            logging.warning(f"Using hosted_vllm model '{self.model_name}' without api_base; defaulting to {self.api_base}")

    def get_completion_kwargs(self, overrides: dict[str, Any] = None) -> dict[str, Any]:
        """Get all kwargs for litellm.completion().

        Returns:
            Dictionary with model, api_key, api_base, and generation_kwargs merged.
        """
        kwargs = self.generation_kwargs.copy()
        if overrides:
            kwargs.update(overrides)

        if self.api_base is not None:
            kwargs["api_base"] = self.api_base

        return kwargs
