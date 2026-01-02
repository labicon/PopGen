import json
import argparse
import os
from openai import OpenAI
import prompt.point_mass_rendezvous as prompt_data

POLICY_GENERATION_PROMPT = """
You are an expert AI agent specializing in generating effective policies for multi-agent coordination tasks.
The behavior will be given to you in JSON format with the following structure:
{{
    "name": str,
    "description": str
}}

Your task is to generate a policy that aligns with the given behavior description.
The policy should be provided as Python code that can be executed in a simulation environment.
The policy should define a function `agent1_policy(obs)` that takes in the current observation of the agent and returns an action.
The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.

The observation `obs` passed to the function is a torch tensor of shape (observation_dim,).
Exact information about the observation format can be found in observation function of the environment.
Observation function of the environment:
{OBSERVATION_FORMAT}

Please provide ONLY the python code for the function `agent1_policy`. 
Do not include markdown formatting or explanations.
Only import torch before function definition if necessary. Do not include any other imports.
"""

def generate_policy(client, behavior, observation_format, previous_policy=None, error_message=None):
    prompt = POLICY_GENERATION_PROMPT.format(OBSERVATION_FORMAT=observation_format)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(behavior)}
    ]
    
    if previous_policy and error_message:
        refinement_message = (
            f"The previous policy generated produced the following error:\n"
            f"{error_message}\n\n"
            f"Here is the previous policy code:\n"
            f"```python\n{previous_policy}\n```\n\n"
            f"Please fix the error and provide the corrected policy code."
        )
        messages.append({"role": "user", "content": refinement_message})

    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages
    )
    
    code = response.choices[0].message.content
    # Clean up markdown code blocks if present
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip(), messages

def save_policy_to_file(policy_code, save_file):
    with open(save_file, "w") as f:
        f.write(policy_code)
    print(f"Saved policy to {save_file}")

def generate_and_save_policy(client: OpenAI, behavior_path, behavior_index, policy_path, previous_policy = None, error_message = None):
    # Load behavior
    if os.path.isdir(behavior_path):
        behavior_file = os.path.join(behavior_path, "behaviors.json")
    else:
        behavior_file = behavior_path

    with open(behavior_file, "r") as f:
        behaviors = json.load(f)
    
    if behavior_index >= len(behaviors):
        print(f"Error: Index {behavior_index} out of range. Only {len(behaviors)} behaviors found.")
        return

    behavior = behaviors[behavior_index]
    print(f"Generating policy for behavior: {behavior['name']}")
    print(f"Description: {behavior['description']}")
    
    policy_code, messages = generate_policy(client, behavior, prompt_data.OBSERVATION_FORMAT, previous_policy, error_message)
    
    print("Generated Policy Code:")
    print(policy_code)
    
    save_policy_to_file(policy_code, policy_path)
    return messages, policy_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate policy from behavior description.")
    parser.add_argument("--input", type=str, required=True, help="Path to behaviors json file")
    parser.add_argument("--index", type=int, default=0, help="Index of behavior to use")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated policy file")
    args = parser.parse_args()

    generate_and_save_policy(args.input, args.index, args.output)
