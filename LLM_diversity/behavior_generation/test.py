"""Main orchestrator for the agentic framework experiments."""

from __future__ import annotations
import json
import argparse
import os
from typing import List, Any

from behavior_generation import (
    InContextStrategy,
    StratifiedStrategy
)
import prompt.point_mass_rendezvous as prompt
import LLM_diversity.cfg.config as config
from openai import OpenAI

def run_in_context(client: OpenAI, output_file: str):
    print(f"\n--- Running In-Context Learning Experiment ---")
    strategy = InContextStrategy()
    behaviors = []
    # Using 5 as per original main_in_context.py
    count = 5 
    for i in range(count):
        print(f"Generating behavior {i+1}/{count}...")
        new_behavior = strategy.generate(client, prompt.ENVIRONMENT_DESCRIPTION)
        behaviors.append(new_behavior)
        print(f" -> {new_behavior}")
    
    save_json(behaviors, output_file)

def run_stratified(client: OpenAI, output_file: str):
    print(f"\n--- Running Stratified Strategy Experiment ---")
    strategy = StratifiedStrategy()
    
    print("Generating behaviors...")
    behaviors = strategy.generate(client, prompt.ENVIRONMENT_DESCRIPTION)
    
    for b in behaviors:
        print(f" -> {b}")
        
    save_json(behaviors, output_file)

def save_json(data: Any, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {filepath}")

def get_output_path(base_path: str, run_idx: int, total_runs: int) -> str:
    if total_runs <= 1:
        return base_path
    
    root, ext = os.path.splitext(base_path)
    return f"{root}_run_{run_idx+1}{ext}"

def main():
    parser = argparse.ArgumentParser(description="Run behavior generation experiments.")
    parser.add_argument("--mode", type=str, 
                        choices=["all", "in_context", "stratified"], 
                        default="all", help="Which experiment to run")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
    args = parser.parse_args()

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n=== Run {i+1}/{args.runs} ===")

        if args.mode in ["all", "in_context"]:
            out = get_output_path(config.IN_CONTEXT_BEHAVIOR_OUTPUT, i, args.runs)
            run_in_context(client, out)
            
        if args.mode in ["all", "stratified"]:
            out = get_output_path(config.SIMPLESTRAT_BEHAVIOR_OUTPUT, i, args.runs)
            run_stratified(client, out)

if __name__ == "__main__":
    main()
