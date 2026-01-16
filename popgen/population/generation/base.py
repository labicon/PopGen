from abc import ABC, abstractmethod

from pqdm.threads import pqdm


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

    def batch_generate(
        self,
        behaviors: list,
        feedbacks_list: list = None,
        num_workers: int = 1,
        **kwargs,
    ) -> list:
        """Generate items for a batch of behaviors.

        Args:
            behaviors: List of behavior descriptions
            feedbacks_list: Optional list of feedbacks per behavior
            num_workers: Number of parallel workers
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of generated items
        """
        if feedbacks_list is None:
            feedbacks_list = [[] for _ in behaviors]

        inputs = [{"behavior": b, "feedbacks": fb, **kwargs} for b, fb in zip(behaviors, feedbacks_list)]

        return pqdm(
            inputs,
            lambda x: self.generate(**x),
            n_jobs=num_workers,
            desc="Generating policies",
        )
