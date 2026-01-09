import json
import argparse
import os
from openai import OpenAI
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
The policy should define a function `agent0_policy(obs)` that takes in the current observation of the agent and returns an action.
The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.

The observation `obs` passed to the function is a torch tensor of shape (observation_dim,).
Exact information about the observation format can be found in observation function of the environment.
Observation function of the environment:
{OBSERVATION_FORMAT}

Please provide ONLY the python code for the function `agent0_policy`. 
Do not include markdown formatting or explanations.
Only import torch before function definition if necessary. Do not include any other imports.
"""

def generate_policy(client, behavior, observation_format, previous_policy=None, error_message=None):
    prompt = POLICY_GENERATION_PROMPT.format(OBSERVATION_FORMAT=observation_format)
    
    messages = [
        {"role": "developer", "content": prompt},
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

def generate_agent1_policy(client, agent0_behavior, agent0_code, observation_format):
    AGENT1_POLICY_GENERATION_PROMPT = """
You are an expert AI agent specializing in generating effective policies for multi-agent coordination tasks.
Your goal is to generate a policy for Agent 1 that effectively coordinates with Agent 0.

You are provided with Agent 0's behavior description and its policy code.

Your task is to generate a policy for Agent 1 (`agent1_policy`) that coordinate with Agent 0's behavior to achieve the task.
The task is reaching a same landmark together in a 2D space.

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
    developer_prompt = AGENT1_POLICY_GENERATION_PROMPT.format(
        OBSERVATION_FORMAT=observation_format
    )

    user_prompt = """Agent 0 Behavior:
{AGENT0_BEHAVIOR}

Agent 0 Policy Code:
```python
{AGENT0_CODE}
```
"""
    user_prompt = user_prompt.format(
        AGENT0_BEHAVIOR=json.dumps(agent0_behavior, indent=2),
        AGENT0_CODE=agent0_code
    )
    
    messages = [
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content":  user_prompt}
    ]
    
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
    agent1_policy_code, agent1_messages = generate_agent1_policy(
        client, behavior, policy_code, prompt_data.OBSERVATION_FORMAT)

    # print("Generated Policy Code:")
    # print(policy_code)
    
    # save_policy_to_file(policy_code, policy_path)
    # return messages, policy_code

    full_policy_code = policy_code + "\n\n" + agent1_policy_code
    save_policy_to_file(full_policy_code, policy_path)
    return messages + agent1_messages, full_policy_code

if __name__ == "__main__":
    behavior_path = "results/2026-01-03_13-07-46/behavior/simple_strat/behaviors.json"
    behavior_index = 2
    policy_path = "policy_rollout/envs/agent0_policy.py"
    client = OpenAI(api_key="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")

    previous_policy = """
import torch

def agent0_policy(obs):
    agent_pos = obs[0:2]
    agent_vel = obs[2:4]
    land_rel_flat = obs[4:-2]
    other_rel = obs[-2:]

    n_landmarks = land_rel_flat.numel() // 2
    if n_landmarks == 0:
        return torch.zeros(1, 2, device=obs.device, dtype=obs.dtype)

    land_rel = land_rel_flat.view(n_landmarks, 2)
    land_abs = land_rel + agent_pos.unsqueeze(0)

    x_coords = land_abs[:, 0]
    min_x = x_coords.min()
    tol = 1e-6
    candidate_mask = (x_coords - min_x).abs() <= tol

    if candidate_mask.any():
        midpoint = agent_pos + 0.5 * other_rel
        dists_sq = ((land_abs - midpoint.unsqueeze(0)) ** 2).sum(dim=1)
        dists_sq_masked = dists_sq.masked_fill(~candidate_mask, float('inf'))
        target_idx = int(torch.argmin(dists_sq_masked))
    else:
        target_idx = int(torch.argmin(x_coords))

    target_pos = land_abs[target_idx]
    kp = 1.5
    kd = 0.5
    force = kp * (target_pos - agent_pos) - kd * agent_vel

    norm = torch.norm(force)
    if torch.isfinite(norm) and norm > 1.0:
        force = force / norm
    force = torch.clamp(force, -1.0, 1.0)

    return force.view(1, 2)
    """
    error_message = """
According to the rule, agent 0 should choose the landmark with the largest x-coordinate. Here, landmark0 at (1.0, 0.0) is uniquely the rightmost, so the expected behavior is to move toward it (no midpoint tie-breaking needed). Instead, agent 0\u2019s x-position steadily decreases from -0.09 to about -0.90, moving left toward landmark2 at (-1.0, 0.0). The images also show the red agent approaching the middle-left green landmark rather than the rightmost one. Therefore, agent 0 did not follow the Rightmost-ClosestMidpoint behavior.
"""

    generate_and_save_policy(client, behavior_path, behavior_index, policy_path, previous_policy, error_message)
