from abc import ABC, abstractmethod
from typing import Any, List, Dict
from pydantic import BaseModel


class Behavior(BaseModel):
    """A single behavior with name and description."""
    name: str
    description: str


class BehaviorList(BaseModel):
    """A list of behaviors."""
    behaviors: List[Behavior]

    def __str__(self) -> str:
        return "\n".join(f"- {b.name}: {b.description}" for b in self.behaviors)


class BehaviorGenerationStrategy(ABC):
    """Abstract base class for behavior generation strategies."""
    def __init__(self, model: str = "openai/gpt-5"):
        super().__init__()
        self.behavior_list: List[Behavior] = []
        self.model = model

    @abstractmethod
    def generate(self, environment_description: str, **kwargs: Any) -> List[Behavior]:
        """
        Generate behaviors based on the environment description.

        Args:
            environment_description: A description of the environment.
            **kwargs: Additional arguments specific to the strategy.

        Returns:
            The generated behavior(s).
        """
        pass
