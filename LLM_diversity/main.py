import os
import sys
import json
import subprocess
import importlib.util
from openai import OpenAI
import cfg.cfg_point_mass_rendezvous as config
from behavior_generation import InContextStrategy, StratifiedStrategy
from policy_rollout.policy_generation import generate_and_save_policy
from behavior_evaluation import BehaviorEvaluator

def generate_policy_rollout_evaluate(client, behavior, behavior_index, prompt_data):
    """
    Generates policy, runs rollout, and evaluates the result.
    Returns (success, result) tuple.
    """
    # 2. Generate Policy and 3. Rollout with Retry
    print("\n--- Generating Policy and Running Rollout (with Retry) ---")
    
    max_retries = 3
    rollout_success = False
    test_output_dir = os.path.join(config.ROLLOUT_OUTPUT, f"behavior_{behavior_index}")

    for attempt in range(max_retries):
        print(f"\nRollout Attempt {attempt + 1}/{max_retries}")
        
        # Generate Policy
        try:            
            generate_and_save_policy(client, behavior_path=config.IN_CONTEXT_BEHAVIOR_OUTPUT, behavior_index=behavior_index, policy_path=config.POLICY_FILE_PATH)
            
            with open(config.POLICY_FILE_PATH, "r") as f:
                policy_code = f.read()
            print("Generated Policy Code (snippet):")
            print(policy_code[:200] + "...")
            
            # Save policy code for logging
            os.makedirs(config.POLICY_OUTPUT, exist_ok=True)
            with open(os.path.join(config.POLICY_OUTPUT, f"policy_{behavior_index}.py"), "w") as f:
                f.write(policy_code)
            print(f"Saved policy code to {os.path.join(config.POLICY_OUTPUT, f'policy_{behavior_index}.py')}")
            
        except Exception as e:
            print(f"Policy generation failed: {e}")
            if attempt == max_retries - 1:
                return False, {"reasoning": f"Policy generation failed after {max_retries} attempts: {e}"}
            continue

        # Run Rollout
        rollout_script_path = config.ROLLOUT_SOURCE_FILE
        if not os.path.exists(rollout_script_path):
            print(f"Error in rollout script path: {rollout_script_path} not found.")
            return False, {"reasoning": "Rollout script not found"}

        try:
            print(f"Running rollout script: {rollout_script_path}")
            
            # Import the module
            # We need to reload the module to pick up the new policy if it was regenerated
            if "rollout_module" in sys.modules:
                del sys.modules["rollout_module"]
                
            spec = importlib.util.spec_from_file_location("rollout_module", rollout_script_path)
            rollout_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rollout_module)
            
            # Call the rollout function
            rollout_module.rollout_behavior(save_path=test_output_dir)
            
            print("Rollout completed successfully.")
            rollout_success = True
            break # Success, exit retry loop
            
        except Exception as e:
            print(f"Rollout failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached for rollout. Aborting.")
                return False, {"reasoning": f"Rollout failed after {max_retries} attempts: {e}"}
            print("Retrying policy generation...")
    
    if not rollout_success:
        return False, {"reasoning": "Rollout failed"}

    # 4. Evaluate
    print("\n--- Evaluating Behavior ---")
    evaluator = BehaviorEvaluator()
    trajectory_path = os.path.join(test_output_dir, "trajectory.txt")
    
    if os.path.exists(trajectory_path):
        with open(trajectory_path, "r") as f:
            trajectory = f.read()
            
        # Evaluate
        print("Running evaluation...")
        result = evaluator.evaluate_trajectory_and_images(
            client,
            trajectory,
            [test_output_dir],
            behavior
        )
        print("Evaluation Result:")
        print(json.dumps(result, indent=2))
        
        return result.get("evaluation", False), result
    else:
        print(f"Error: Trajectory file not found at {trajectory_path}")
        return False, {"reasoning": "Trajectory file not found"}

def test_full_pipeline():
    # Load prompt data dynamically
    prompt_file_path = os.path.abspath(config.ENVIRONMENT_PROMPT_FILE)
    spec = importlib.util.spec_from_file_location("prompt_data", prompt_file_path)
    prompt_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_data)

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # 1. Generate Behavior
    print("--- 1. Generating Behavior ---")
    # strategy = InContextStrategy()
    strategy = StratifiedStrategy()
    # Mocking previous preferences or just generating one
    # The generate method signature is generate(client, environment_description, **kwargs)
    # InContextStrategy.generate takes (client, environment_description)
    try:
        behaviors = strategy.generate(client, prompt_data.ENVIRONMENT_DESCRIPTION)
        print(f"Generated Behaviors: {json.dumps(behaviors, indent=2)}")

        # Save behavior to file
        os.makedirs(os.path.dirname(config.IN_CONTEXT_BEHAVIOR_OUTPUT), exist_ok=True)
        behavior_file_path = os.path.join(config.IN_CONTEXT_BEHAVIOR_OUTPUT, "behaviors.json")
        with open(behavior_file_path, "w") as f:
            json.dump(behaviors, f, indent=2)
        print(f"Saved generated behaviors to {behavior_file_path}")

    except Exception as e:
        print(f"Behavior generation failed: {e}")
        return
    
    if not behaviors:
        print("No behaviors generated.")
        return

    # Loop over each behavior
    for idx, behavior in enumerate(behaviors):
        print(f"\n--------------------------------------------------")
        print(f"Processing Behavior {idx + 1}/{len(behaviors)}: {behavior.get('name', 'Unknown')}")
        print(f"--------------------------------------------------")
        
        # Loop until policy is aligned with behavior
        max_alignment_retries = 5
        for i in range(max_alignment_retries):
            print(f"\n=== Alignment Attempt {i+1}/{max_alignment_retries} ===")
            success, result = generate_policy_rollout_evaluate(client, behavior, idx, prompt_data)
            
            if success:
                print("\nSUCCESS: Policy aligns with behavior description!")
                break
            else:
                print(f"\nFAILURE: Policy does not align. Reasoning: {result.get('reasoning', 'Unknown')}")
                if i == max_alignment_retries - 1:
                    print("Max alignment retries reached. Failed to generate aligned policy.")

if __name__ == "__main__":
    test_full_pipeline()
