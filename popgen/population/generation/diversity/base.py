from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from popgen.utils.config import LitellmConfig

T = TypeVar("T", bound=BaseModel)


class GenerationStrategy(ABC):
    """Abstract base class for diverse generation strategies.

    This class is generic and can generate any Pydantic model type.
    The output type is specified via constructor arguments.
    """

    def __init__(self, model_config: LitellmConfig | dict, output_type: type[T], output_dir: str | None = None):
        super().__init__()

        if not isinstance(model_config, LitellmConfig):
            model_config = LitellmConfig(**model_config)

        self.model_config = model_config
        self.output_type = output_type
        self.output_dir = output_dir

        self.generated_items: list[T] = []

    @abstractmethod
    def generate(self, input_prompt: str, **kwargs: Any) -> list[T]:
        """
        Generate items based on the input prompt.

        Args:
            input_prompt: A prompt describing the generation task.
            **kwargs: Additional arguments specific to the strategy.

        Returns:
            The generated item(s).
        """
        pass

    def _list_to_string(self, items: list[T]) -> str:
        """Format a list of items as a string for context."""
        if not items:
            return "None"
        return "\n".join(f"- {item if isinstance(item, str) else item.model_dump_json()}" for item in items)
