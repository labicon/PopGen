from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class GenerationStrategy(ABC):
    """Abstract base class for diverse generation strategies.

    This class is generic and can generate any Pydantic model type.
    The output type is specified via constructor arguments.
    """

    def __init__(self, model: str, output_type: type[T]):
        super().__init__()
        self.generated_items: list[T] = []
        self.model = model
        self.output_type = output_type

    @abstractmethod
    def generate(self, environment_description: str, **kwargs: Any) -> list[T]:
        """
        Generate items based on the environment description.

        Args:
            environment_description: A description of the environment.
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
