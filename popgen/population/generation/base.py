from abc import ABC, abstractmethod

import litellm
from pqdm.threads import pqdm
from pydantic import BaseModel

from popgen.utils.config import LitellmConfig


class BaseGenerator(ABC):
    """Abstract base class for behavior-based generation."""

    def __init__(self, generation_type: str):
        self.generation_type = generation_type

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
    """Abstract base class for generators that utilize feedback.

    Supports both string-based model names (backward compatible) and
    LitellmConfig objects for more advanced configuration.

    Example:
        >>> # Simple usage with model name
        >>> generator = MyGenerator(model="openai/gpt-4o-mini")
        >>>
        >>> # Advanced usage with config
        >>> from popgen.utils.config import LitellmConfig
        >>> config = LitellmConfig(
        ...     model_name="hosted_vllm/Qwen/Qwen3-4B-Instruct",
        ...     api_base="http://localhost:8000/v1",
        ...     generation_kwargs={"temperature": 0.7}
        ... )
        >>> generator = MyGenerator(model=config)
    """

    def __init__(self, model_config: LitellmConfig, generation_type: str):
        super().__init__(generation_type=generation_type)
        self.model_config = model_config
        self.model_name = model_config.model_name

    def generate(self, user_prompt: str = None, system_prompt: str = None, feedbacks: list = None, **kwargs) -> dict | BaseModel:
        """Generate items based on prompts and feedback.

        Args:
            user_prompt: The user prompt for generation.
            system_prompt: The system prompt for generation.
            feedbacks: Optional feedbacks to refine generation.
            **kwargs: Additional arguments passed to litellm.completion().
                These override any generation_kwargs from the config.

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
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.get_completion_kwargs(overrides=kwargs),
        )

        return response
