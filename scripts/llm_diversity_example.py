"""Example script demonstrating the diversity discriminator.

This script shows how to use the DiversityDiscriminator to rate how diverse
each text (in this case, behavior descriptions) is compared to all others.
"""

import json
import os
from typing import Any

from popgen.population.evaluation import DiversityDiscriminator
from popgen.utils.config import LitellmConfig


def main():
    """Example usage of DiversityDiscriminator."""

    # Example behaviors - could be loaded from a file
    behaviors = [
        "Aggressive forward player who constantly pushes toward the goal and takes many shots.",
        "Defensive player who stays back and focuses on blocking opponent attacks.",
        "Passing specialist who prioritizes ball distribution and creating opportunities for teammates.",
        "Ball hog who rarely passes and tries to score alone.",
        "Team player who supports others and maintains good positioning.",
    ]

    print("=" * 80)
    print("Diversity Discriminator Example")
    print("=" * 80)
    print(f"\nEvaluating {len(behaviors)} behaviors:")
    for i, behavior in enumerate(behaviors, 1):
        print(f"  {i}. {behavior[:60]}...")

    # Initialize the evaluator
    # You can use any LiteLLM-compatible model string
    judge_model = "openai/gpt-4o-mini"  # or "anthropic/claude-3-sonnet-20240229", etc.

    print(f"\nInitializing DiversityDiscriminator with model: {judge_model}")
    model_config = LitellmConfig(model_name=judge_model)
    evaluator = DiversityDiscriminator(
        model_config=model_config,
        num_workers=4,  # Parallel workers for evaluation
    )

    # Evaluate the behaviors
    print("\nEvaluating diversity (each text vs. all others)...")
    print("This will make N LLM calls (one per text)")

    result = evaluator.evaluate(
        inputs=behaviors,
        metadata={
            "environment": "example",
            "description": "Sample text diversity evaluation",
        }
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nInstance-level results (per text):")
    print("-" * 80)
    for i, metric in enumerate(result.instance_metrics):
        print(f"\nText {i+1}:")
        print(f"  Content: {metric['target_text'][:70]}...")
        print(f"  Diversity Score: {metric['diversity_score']}/5")
        print(f"  Reasoning: {metric['reasoning'][:150]}...")
        if metric['error']:
            print(f"  Error: {metric['error']}")

    print("\n" + "=" * 80)
    print("Overall metrics (aggregated):")
    print("-" * 80)
    for metric_name, stats in result.overall_metrics.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"{metric_name}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std:  {stats['std']:.2f}")
            print(f"  Min:  {stats['min']:.2f}")
            print(f"  Max:  {stats['max']:.2f}")
        else:
            print(f"{metric_name}: {stats}")

    # Save results to file
    output_dir = "example_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "llm_diversity_results.json")

    evaluator.save_results(result, output_path)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
