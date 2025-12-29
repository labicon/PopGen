from openai import OpenAI
from typing import List, Any, Dict
from .base import BehaviorGenerationStrategy

class InContextStrategy(BehaviorGenerationStrategy):
    def __init__(self):
        super().__init__()

    def generate(self, client: OpenAI, environment_description: str, **kwargs: Any) -> List[Dict[str, str]]:
        """
        Generate 4 behaviors in an in-context manner, using previously generated behaviors as context.
        """
        behaviors = []
        for _ in range(4):
            behavior = self.generate_single(client, environment_description)
            behaviors.append(behavior)
        return behaviors

    def generate_single(self, client: OpenAI, environment_description: str) -> str:
        """
        Generate a single behavior preference, taking into account previous preferences.
        
        Kwargs:
            previous_preferences (list[str]): List of previously generated behaviors.
        """
        previous_preferences = self.behavior_list
        
        prompt_format = {
            "id": "pmpt_6941f01a832481968d266825c7884e6503aa32b03f82dea9",
            "variables": {
                "environment_description": environment_description,
                "previous_preferences": self.behavior_list_to_string(previous_preferences)
            }
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.responses.create(
                    model="gpt-5",
                    prompt=prompt_format,
                    text = self.behavior_format
                )

                print(f"DEBUG: behavior_generator output:\n{response.output_text}\n")

                behavior = self.parse_text_to_dict(response.output_text)
                self.behavior_list.append(behavior)

                return behavior
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise

