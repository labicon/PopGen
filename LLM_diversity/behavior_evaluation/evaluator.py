from openai import OpenAI
from typing import Any, Dict, List, Optional, Union
import json
import base64
import os

class BehaviorEvaluator:
    def __init__(self):
        self.evaluation_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "behavior_evaluation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "evaluation": {"type": "boolean", "description": "A true or false evaluation that indicates whether the behavior was followed."},
                        "reasoning": {"type": "string", "description": "Explanation for the score."}
                    },
                    "required": ["evaluation", "reasoning"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def evaluate_trajectory(self, client: OpenAI, trajectory: str, behavior_description: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate agent behavior based on trajectory data.
        
        Args:
            client: The OpenAI client.
            trajectory: String representation of the trajectory.
            behavior_description: Dictionary containing 'name' and 'description' of the expected behavior.
            
        Returns:
            Dict containing 'evaluation' and 'reasoning'.
        """
        developer_prompt = f"""
You are an expert in evaluating agent behaviors.
You will be provided with a behavior description and trajectory data.
Your task is to evaluate how well the agent followed the behavior description based on the trajectory data.

The output should be a JSON object with the following format:
{{
    "evaluation": true or false,
    "reasoning": "Explanation for the evaluation."
}}
"""
        user_prompt = f"""
Behavior Name: {behavior_description.get('name', 'Unknown')}
Behavior Description: {behavior_description.get('description', '')}
Trajectory Data:
{trajectory}

Evaluate if the agent followed the behavior description based on the trajectory data.
"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": developer_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=self.evaluation_format
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Evaluation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {"evaluation": False, "reasoning": f"Evaluation failed: {e}"}

    def evaluate_trajectory_and_images(self, client: OpenAI, trajectory: str, image_paths: List[str], behavior_description: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate agent behavior based on trajectory data and sequential images.
        
        Args:
            client: The OpenAI client.
            trajectory: String representation of the trajectory.
            image_paths: List of paths to the image files (sequential frames) or folders containing sequential images.
            behavior_description: Dictionary containing 'name' and 'description' of the expected behavior.
            
        Returns:
            Dict containing 'evaluation' and 'reasoning'.
        """
        # Encode images
        encoded_images = []
        try:
            for image_path in image_paths:
                encoded = encode_image(image_path)
                if isinstance(encoded, list):
                    encoded_images.extend(encoded)
                else:
                    encoded_images.append(encoded)
        except Exception as e:
             return {"evaluation": False, "reasoning": f"Failed to read images: {e}"}

        developer_prompt = f"""
You are an expert in evaluating agent behaviors using both trajectory data and visual information.
You will be provided with a behavior description, trajectory data, and a sequence of images from the rollout.
Your task is to evaluate how well the agent followed the behavior description based on the trajectory data and the provided images.

The output should be a JSON object with the following format:
{{
    "evaluation": true or false,
    "reasoning": "Explanation for the evaluation."
}}
"""
        user_prompt = f"""
Behavior Name: {behavior_description.get('name', 'Unknown')}
Behavior Description: {behavior_description.get('description', '')}
Trajectory Data:
{trajectory}

Evaluate how well the agent followed the behavior description based on the trajectory data and the provided sequential images.
Provide your evaluation and reasoning.
"""
        
        content = [{"type": "text", "text": user_prompt}]
        for encoded_string in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_string}"
                }
            })

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": developer_prompt},
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    response_format=self.evaluation_format
                )

                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Evaluation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {"evaluation": False, "reasoning": f"Evaluation failed: {e}"}

# Function to encode the image
def encode_image(image_path: str) -> Union[str, List[str]]:
    if os.path.isdir(image_path):
        encoded_images = []
        # List files matching image_*.png
        files = [f for f in os.listdir(image_path) if f.startswith("image_") and f.endswith(".png")]
        # Sort by the number in the filename
        files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
        
        for f in files:
            file_path = os.path.join(image_path, f)
            with open(file_path, "rb") as image_file:
                encoded_images.append(base64.b64encode(image_file.read()).decode("utf-8"))
        return encoded_images
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
if __name__ == "__main__":
    import sys
    # Add parent directory to path to import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import LLM_diversity.cfg.config as config

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    evaluator = BehaviorEvaluator()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "results", "rollout", "test")
    
    trajectory_path = os.path.join(data_path, "trajectory.txt")
    if os.path.exists(trajectory_path):
        with open(trajectory_path, "r") as f:
            trajectory = f.read()

        behavior_description = {
            "name": "Stay Still",
            "description": "The agents should stay at their initial positions and not move."
        }

        print("Evaluating...")
        result = evaluator.evaluate_trajectory_and_images(
            client, 
            trajectory, 
            [data_path], # Pass folder path
            behavior_description
        )
        print(json.dumps(result, indent=2))
    else:
        print(f"Trajectory file not found at {trajectory_path}")
