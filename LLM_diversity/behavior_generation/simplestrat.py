from openai import OpenAI
from typing import List, Any, Dict
from .base import BehaviorGenerationStrategy
import json

class StratifiedStrategy(BehaviorGenerationStrategy):
    def __init__(self):
        super().__init__()
        # In this strategy, the chat output should be list of behaviors
        self.behavior_format = {
            "format": {
                "type": "json_schema",
                "name": "player_behaviors",
                "schema": {
                    "type": "object",
                    "properties": {
                        "behaviors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["name", "description"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["behaviors"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def generate(self, client: OpenAI, environment_description: str, **kwargs: Any) -> List[Dict[str, str]]:
        """
        Generate behaviors by first identifying strategic dividers and then sampling from partitions.
        """
        # 1. Generate Divider 1
        divider1 = self._generate_single_divider(client, environment_description, [])
        
        # 2. Generate Divider 2
        divider2 = self._generate_single_divider(client, environment_description, [divider1])
        
        # 3. Generate Behaviors based on dividers
        behaviors = self._generate_behaviors_from_dividers(client, environment_description, divider1, divider2)
                
        return behaviors

    def _generate_single_divider(self, client: OpenAI, environment_description: str, existing_dividers: List[str]) -> str:    
        prompt_format = {
            "id": "pmpt_6941f3b1056881958342b62e873c4d80030aadf39e271cfc",
            "variables": {
                "environment_description": environment_description,
                "existing_dividers": self.list_to_string(existing_dividers)
            }
        }
        
        response = client.responses.create(
            model="gpt-5",
            prompt=prompt_format,
        )
        print(f"DEBUG: simple_strat divider output:\n{response.output_text}\n")
        return response.output_text.strip()

    def _generate_behaviors_from_dividers(self, client: OpenAI, environment_description: str, divider1: str, divider2: str) -> List[Dict[str, str]]:
        prompt_format = {
            "id": "pmpt_6941fa219c308196afbbb153616aa44c0a2d58b38ce9e342",
            "variables": {
                "environment_description": environment_description,
                "dividers": self.list_to_string([divider1, divider2])
            }
        }
        
        response = client.responses.create(
            model="gpt-5",
            prompt=prompt_format,
            max_output_tokens=10000,
            text=self.behavior_format
        )
        print(f"DEBUG: simple_strat response output:\n{response}\n")
        print(f"DEBUG: simple_strat behavior output:\n{response.output_text}\n")
        
        # Extract behaviors
        behaviors = self.parse_text_to_list_dict(response.output_text)
        self.behavior_list.extend(behaviors)
                
        return behaviors

    def list_to_string(self, items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items)
    
    def parse_text_to_list_dict(self, text: str) -> List[Dict[str, str]]:
        """Parse the output text into a list of behavior dictionaries."""
        try:
            data = json.loads(text)
            behavior_list = data.get("behaviors", [])
            for behavior in behavior_list:
                if not all(key in behavior for key in ["name", "description"]):
                    raise ValueError("Parsed JSON does not contain required keys.")
            return behavior_list
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")