from typing import List, Any
import litellm
from .base import BehaviorGenerationStrategy, Behavior, BehaviorList

BEHAVIOR_GENERATION_PROMPT = """You are designing diverse player behaviors for a multi-agent environment.

Environment Description:
{environment_description}

Previously generated behaviors:
{previous_preferences}

Generate a NEW and UNIQUE player behavior that is distinctly different from the previously generated ones. The behavior should represent a coherent strategy or personality that a player might adopt in this environment.

Focus on creating diversity - the new behavior should explore a different aspect of the strategy space."""


class InContextStrategy(BehaviorGenerationStrategy):
    def __init__(self, model: str = "openai/gpt-5"):
        super().__init__(model)

    def generate(self, environment_description: str, **kwargs: Any) -> List[Behavior]:
        """
        Generate behaviors in an in-context manner, using previously generated behaviors as context.
        """
        behaviors = []
        num_behaviors = kwargs.get("num_behaviors", 5)
        for _ in range(num_behaviors):
            behavior = self.generate_single(environment_description)
            behaviors.append(behavior)
        return behaviors

    def generate_single(self, environment_description: str) -> Behavior:
        """
        Generate a single behavior preference, taking into account previous preferences.
        """
        previous_preferences = BehaviorList(behaviors=self.behavior_list)

        prompt = BEHAVIOR_GENERATION_PROMPT.format(
            environment_description=environment_description,
            previous_preferences=str(previous_preferences)
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=Behavior,
                )

                behavior = Behavior.model_validate_json(response.choices[0].message.content)
                self.behavior_list.append(behavior)

                return behavior
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
