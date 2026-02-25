from abc import ABC, abstractmethod

import litellm
from pqdm.threads import pqdm
from pydantic import BaseModel

from popgen.utils.config import LitellmConfig
from popgen.utils.decorators import track_time


class BaseGenerator(ABC):
    """Abstract base class for behavior-based generation."""

    def __init__(self, generation_type: str):
        self.generation_type = generation_type

    @property
    def metadata(self) -> dict:
        return {"generation_type": self.generation_type}

    @abstractmethod
    def generate(self, *args, **kwargs) -> dict | BaseModel:
        """Generate items based on the given behavior description."""
        pass

    def batch_generate(
        self,
        inputs: list[dict],
        num_workers: int = 1,
        timed: bool = False,
        **kwargs,
    ) -> tuple[list, list[float]]:
        """Generate items for a batch of behaviors.

        Args:
            inputs: List of per-item kwargs dicts passed to generate()
            num_workers: Number of parallel workers
            timed: Whether to record per-item wall-clock latencies
            **kwargs: Additional arguments merged into every input

        Returns:
            Tuple of (results, latencies) where latencies[i] is the elapsed
            wall-clock seconds for item i (0.0 if timed=False).
        """
        inputs = [{**inp, **kwargs} for inp in inputs]

        if timed:
            worker = track_time("generate")(lambda x: self.generate(**x))
        else:

            def worker(x):
                return self.generate(**x)

        raw_results = pqdm(
            inputs,
            worker,
            n_jobs=num_workers,
            desc="Generating policies",
        )

        results, latencies = [], []
        for item in raw_results:
            if isinstance(item, Exception):
                results.append(item)
                latencies.append(0.0)
            elif timed:
                results.append(item[0])
                latencies.append(item[1])
            else:
                results.append(item)
                latencies.append(0.0)
        return results, latencies


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

    @property
    def metadata(self) -> dict:
        return {"generation_type": self.generation_type, "model_name": self.model_name}

    def generate(self, user_prompt: str, system_prompt: str = None, examples: list | None = None, feedbacks: list = None, **kwargs) -> dict | BaseModel:
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

        formatted_prompt = ""

        # Incorporate example into the prompt to provide a clear reference for the model, especially if the generation task is complex or has specific formatting requirements.
        if examples is not None:
            examples_str = "\n\n".join(f"Example {i}:\n{e}" for (i, e) in enumerate(examples))
            formatted_prompt += f"# Examples\n{examples_str}\n\n"

        # Incorporate feedback into the prompt to guide refinement. This is especially useful for iterative generation where the model can learn from previous attempts.
        if feedbacks is not None and len(feedbacks) > 0:
            feedback_previous = "\n\n".join(f" - {self.generation_type} #({i}):\n{e}\nFeedback: {fdb}" for (i, (e, fdb)) in enumerate(feedbacks))
            refinement_message = (
                f"The previous attempts produced the following errors and feedbacks:\n{feedback_previous}\n\nPlease fix the errors and provide the corrected {self.generation_type}.\n\n"
            )
            formatted_prompt += refinement_message

        # Append the user prompt at the end to ensure it is the main focus of the generation, especially after providing examples and feedback.
        formatted_prompt += user_prompt

        messages.append({"role": "user", "content": formatted_prompt})

        response = litellm.completion(
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.get_completion_kwargs(overrides=kwargs),
        )

        return response
