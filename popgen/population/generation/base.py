from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base class for behavior-based generation."""

    @property
    def generation_type(self) -> str:
        """Type of items generated (e.g., 'policy_code', 'reward_function')."""
        # name of the class without 'generator' suffix
        return self.__class__.__name__.replace("Generator", "")

    @abstractmethod
    def generate(self, behavior: str, **kwargs) -> dict:
        """Generate items based on the given behavior description."""
        pass
