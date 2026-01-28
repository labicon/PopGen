from abc import ABC, abstractmethod

import litellm
from pqdm.threads import pqdm
from pydantic import BaseModel


class BaseGenerator(ABC):
    """Abstract base class for behavior-based generation."""

    @abstractmethod
    def generate(self, *args, **kwargs) -> dict | BaseModel:
        """Generate items based on the given behavior description."""
        pass

    def batch_generate(
        self,
        inputs: list[dict],
        num_workers: int = 1,
        **kwargs,
    ) -> list:
        """Generate items for a batch of behaviors.

        Args:
            *args: Positional arguments passed to generate()
            num_workers: Number of parallel workers
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of generated items
        """
        # merge with kwargs
        inputs = [{**inp, **kwargs} for inp in inputs]
        return pqdm(
            inputs,
            lambda x: self.generate(**x),
            n_jobs=num_workers,
            desc="Generating policies",
        )


class GeneratorWithFeedback(BaseGenerator):
    """Abstract base class for generators that utilize feedback."""

    def __init__(self, model: str, generation_type: str):
        super().__init__()
        self.model = model
        self.generation_type = generation_type

    def generate(self, user_prompt: str = None, system_prompt: str = None, feedbacks: list = None, **kwargs) -> dict | BaseModel:
        """Generate items based on prompts and feedback.

        Args:
            user_prompt: The user prompt for generation.
            system_prompt: The system prompt for generation.
            feedbacks: Optional feedbacks to refine generation.
            **kwargs: Additional arguments passed to generation.
        Returns:
            Generated items as a dict or pydantic.BaseModel.
        """
        messages = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if feedbacks is not None and len(feedbacks) > 0:
            feedback_previous = "\n\n".join(f" - {self.generation_type} #({i}):\n{e}\nFeedback: {fdb}" for (i, (e, fdb)) in enumerate(feedbacks))
            refinement_message = f"The previous attempts produced the following errors and feedbacks:\n{feedback_previous}\n\nPlease fix the errors and provide the corrected {self.generation_type}."
            messages.append({"role": "user", "content": refinement_message})

        response = litellm.completion(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        return response
