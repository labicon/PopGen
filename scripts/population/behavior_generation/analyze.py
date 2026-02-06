"""Analyzer script for evaluating generated behaviors across different methods.

This script:
1. Loads generation results from different method-model pairs
2. Evaluates behaviors using quality and diversity evaluators
3. Aggregates scores by method
4. Plots bar charts for each metric and quality vs diversity scatter plot
"""

import argparse
import importlib
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from popgen.population.evaluation.quality import BehaviorQualityEvaluator
from popgen.population.evaluation.diversity import DiversityEvaluator
from popgen.utils.config import LitellmConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_behaviors_from_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all behaviors from a results directory organized by method-model pairs.

    Args:
        results_dir: Path to the results directory (e.g., results/football/)

    Returns:
        Dictionary mapping method names to lists of behavior dicts
    """
    behaviors_by_method = {}

    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir() or method_dir.name.startswith("."):
            continue

        method_name = method_dir.name
        all_behaviors = []

        # Load behaviors from all groups
        for group_dir in sorted(method_dir.glob("group_*")):
            behaviors_file = group_dir / "behaviors.json"
            if behaviors_file.exists():
                with open(behaviors_file) as f:
                    behaviors = json.load(f)
                    all_behaviors.extend(behaviors)

        if all_behaviors:
            behaviors_by_method[method_name] = all_behaviors
            logger.info(f"Loaded {len(all_behaviors)} behaviors from {method_name}")

    return behaviors_by_method


def evaluate_quality(
    behaviors: list[dict],
    env_description: str,
    judge_model: str,
    num_workers: int = 8,
) -> dict:
    """Evaluate the quality of behaviors.

    Args:
        behaviors: List of behavior dicts with 'name' and 'description' keys
        env_description: Environment description for context
        judge_model: Model to use for quality evaluation
        num_workers: Number of parallel workers

    Returns:
        Dictionary with quality metrics
    """
    judge_config = LitellmConfig(model_name=judge_model)
    evaluator = BehaviorQualityEvaluator(model_config=judge_config, num_workers=num_workers)

    # Prepare inputs for the evaluator
    inputs = []
    for behavior in behaviors:
        behavior_text = f"{behavior['name']}: {behavior['description']}"
        inputs.append({
            "behavior": behavior_text,
            "environment_description": env_description,
        })

    result = evaluator.evaluate(inputs)
    return result.overall_metrics


def evaluate_diversity(
    behaviors: list[dict],
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """Evaluate the diversity of behaviors.

    Args:
        behaviors: List of behavior dicts with 'name' and 'description' keys
        embedding_model: Sentence transformer model for embeddings

    Returns:
        Dictionary with diversity metrics
    """
    evaluator = DiversityEvaluator(embedding_model=embedding_model)

    # Prepare texts for the evaluator
    texts = [f"{b['name']}: {b['description']}" for b in behaviors]
    inputs = [{"texts": texts}]

    result = evaluator.evaluate(inputs)
    return result.overall_metrics


def plot_metric_bars(
    results_df: pd.DataFrame,
    metric_columns: list[str],
    output_path: Path,
    title: str = "Evaluation Metrics by Method",
    palette: str = "Set2",
):
    """Plot bar charts for each metric.

    Args:
        results_df: DataFrame with methods as index and metrics as columns
        metric_columns: List of metric column names to plot
        output_path: Path to save the plot
        title: Plot title
        palette: Matplotlib colormap name
    """
    n_metrics = len(metric_columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    cmap = plt.colormaps.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(results_df)))

    for ax, metric in zip(axes, metric_columns):
        values = results_df[metric].values
        methods = results_df.index.tolist()
        # Format labels as "Strategy\nModel" and remove "Strategy" suffix
        labels = [m.replace("Strategy", "").replace("__", "\n") for m in methods]

        bars = ax.bar(range(len(methods)), values, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add top margin to prevent text from being cut off
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.10)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metric bars plot to {output_path}")


def plot_quality_vs_diversity(
    results_df: pd.DataFrame,
    quality_metric: str,
    diversity_metric: str,
    output_path: Path,
    palette: str = "Set2",
):
    """Plot quality vs diversity scatter plot.

    Args:
        results_df: DataFrame with methods as index and metrics as columns
        quality_metric: Column name for quality metric
        diversity_metric: Column name for diversity metric
        output_path: Path to save the plot
        palette: Matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.colormaps.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(results_df)))

    for i, (method, row) in enumerate(results_df.iterrows()):
        # Format label as "Name\nModel" and remove "Strategy" suffix
        label = method.replace("Strategy", "").replace("__", "\n")
        ax.scatter(
            row[diversity_metric],
            row[quality_metric],
            s=150,
            c=[colors[i]],
            label=label,
            edgecolors="black",
            linewidths=1,
        )

    ax.set_xlabel(diversity_metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(quality_metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Quality vs Diversity Trade-off", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add method labels next to points
    for i, (method, row) in enumerate(results_df.iterrows()):
        # Format annotation as "Name\nModel" and remove "Strategy" suffix
        label = method.replace("Strategy", "").replace("__", "\n")
        ax.annotate(
            label,
            (row[diversity_metric], row[quality_metric]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved quality vs diversity plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generated behaviors using quality and diversity evaluators"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/football",
        help="Path to results directory containing method-model subdirectories",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="football",
        help="Environment name for loading description",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-5",
        help="Model to use for quality evaluation",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Sentence transformer model for diversity evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots and results (defaults to results-dir)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for evaluation",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip quality evaluation (useful for testing)",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="Set2",
        help="Matplotlib colormap name (e.g., Set1, Set2, Set3, tab10, Pastel1, Dark2)",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Load cached results from CSV instead of re-running evaluation",
    )

    args = parser.parse_args()

    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load environment description
    env_module = importlib.import_module(f"popgen.envs.{args.env}")
    env_description = env_module.ENVIRONMENT_DESCRIPTION

    logger.info("=" * 60)
    logger.info("BEHAVIOR ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info("-" * 60)

    # Check for cached results
    csv_path = output_dir / "analysis_results.csv"
    if csv_path.exists():
        logger.info(f"Loading cached results from {csv_path}")
        results_df = pd.read_csv(csv_path, index_col="method")
    else:
        # Load behaviors
        behaviors_by_method = load_behaviors_from_results(results_dir)

        if not behaviors_by_method:
            logger.error("No behaviors found in results directory")
            return

        # Evaluate each method
        all_results = []

        for method_name, behaviors in behaviors_by_method.items():
            logger.info(f"\nEvaluating {method_name}...")

            result = {"method": method_name, "n_behaviors": len(behaviors)}

            # Quality evaluation
            if not args.skip_quality:
                logger.info("  Running quality evaluation...")
                quality_metrics = evaluate_quality(
                    behaviors,
                    env_description,
                    args.judge_model,
                    args.num_workers,
                )
                result["quality_total"] = quality_metrics["total"]["mean"]
                result["quality_relevance"] = quality_metrics["relevance"]["mean"]
                result["quality_realism"] = quality_metrics["realism"]["mean"]
                result["quality_specificity"] = quality_metrics["specificity"]["mean"]
            else:
                result["quality_total"] = 0.0
                result["quality_relevance"] = 0.0
                result["quality_realism"] = 0.0
                result["quality_specificity"] = 0.0

            # Diversity evaluation
            logger.info("  Running diversity evaluation...")
            diversity_metrics = evaluate_diversity(behaviors, args.embedding_model)
            result["diversity_semantic"] = diversity_metrics["semantic_diversity"]["mean"]
            result["diversity_lexical"] = diversity_metrics["lexical_diversity"]["mean"]
            result["vocabulary_richness"] = diversity_metrics["vocabulary_richness"]["mean"]

            all_results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.set_index("method", inplace=True)

        # Save results to CSV
        results_df.to_csv(csv_path)
        logger.info(f"\nSaved results to {csv_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    print(results_df.to_string())

    # Plot metrics
    logger.info("\nGenerating plots...")

    # Quality metrics bar plot
    quality_metrics = ["quality_relevance", "quality_realism", "quality_specificity", "quality_total"]
    if not args.skip_quality:
        plot_metric_bars(
            results_df,
            quality_metrics,
            output_dir / "quality_metrics.png",
            title="Quality Metrics by Method",
            palette=args.palette,
        )

    # Diversity metrics bar plot
    diversity_metrics = ["diversity_semantic", "diversity_lexical", "vocabulary_richness"]
    plot_metric_bars(
        results_df,
        diversity_metrics,
        output_dir / "diversity_metrics.png",
        title="Diversity Metrics by Method",
        palette=args.palette,
    )

    # All metrics bar plot
    all_metrics = ["quality_total", "diversity_semantic", "diversity_lexical"]
    if not args.skip_quality:
        plot_metric_bars(
            results_df,
            all_metrics,
            output_dir / "all_metrics.png",
            title="All Metrics by Method",
            palette=args.palette,
        )

    # Quality vs Diversity scatter plot
    if not args.skip_quality:
        plot_quality_vs_diversity(
            results_df,
            quality_metric="quality_total",
            diversity_metric="diversity_semantic",
            output_path=output_dir / "quality_vs_diversity.png",
            palette=args.palette,
        )

    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
