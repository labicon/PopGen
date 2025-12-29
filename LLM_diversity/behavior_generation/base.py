from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Any, List, Union, Dict
import json

# Answer format
# This json has {"name": str, "description": str}
behavior_format = {
    "format": {
        "type": "json_schema",
        "name": "player_behavior",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["name", "description"],
            "additionalProperties": False
        },
        "strict": True
    }
}

class BehaviorGenerationStrategy(ABC):
    """Abstract base class for behavior generation strategies."""
    def __init__(self):
        super().__init__()
        self.behavior_list = []
        self.behavior_format = behavior_format
    
    @abstractmethod
    def generate(self, client: OpenAI, environment_description: str, **kwargs: Any) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        Generate behaviors based on the environment description.
        
        Args:
            client: The OpenAI client.
            environment_description: A description of the environment.
            **kwargs: Additional arguments specific to the strategy (e.g., previous_preferences).
            
        Returns:
            The generated behavior(s). The format should be a dictionary with keys "name" and "description", or a list of such dictionaries.
        """
        pass

    def parse_text_to_dict(self, text: str) -> Dict[str, str]:
        """Parse the output text into a behavior dictionary."""
        try:
            behavior_dict = json.loads(text)
            if "name" in behavior_dict and "description" in behavior_dict:
                return behavior_dict
            else:
                raise ValueError("Parsed JSON does not contain required keys.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
        
    def behavior_list_to_string(self, behaviors: List[Dict[str, str]]) -> str:
        """Convert a list of behaviors to a formatted string."""
        if not behaviors:
            return "None"
        return "\n".join(f"- {b['name']}: {b['description']}" for b in behaviors)