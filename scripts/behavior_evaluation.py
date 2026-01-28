"""Behavior diversity and quality analysis."""

from __future__ import annotations

from typing import Any, Optional
import importlib
import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from popgen.population.evaluation import BehaviorQualityEvaluator, DiversityEvaluator


def _behaviors_to_texts(behaviors: list[dict[str, Any]]) -> list[str]:
    """Convert behaviors to text representations."""
    return [f"{b.get('name', '')}: {b.get('description', '')}" for b in behaviors]


def plot_behavior_tsne(
    behaviors: list[dict[str, Any]],
    output_path: str,
    embeddings: np.ndarray,
    semantic_similarity_matrix: np.ndarray,
    lexical_similarity_matrix: np.ndarray,
    semantic_diversity: float,
    lexical_diversity: float,
    perplexity: float = 5.0,
    env_name: str = None,
    strategy: str = None,
) -> dict[str, Any]:
    """Visualize behaviors using t-SNE and similarity matrix, with diversity scores.

    Args:
        behaviors: List of dicts with keys `name` and `description`.
        output_path: Directory path to save outputs.
        embeddings: Embedding matrix for behaviors.
        semantic_similarity_matrix: Pairwise semantic similarity matrix for behaviors.
        lexical_similarity_matrix: Pairwise lexical similarity matrix for behaviors.
        semantic_diversity: Semantic diversity score.
        lexical_diversity: Lexical diversity score.
        perplexity: t-SNE perplexity parameter.
        env_name: Optional environment name to display.
        strategy: Optional strategy name to display.

    Returns:
        Dict containing embeddings, similarity matrices, and diversity scores.
    """
    n_behaviors = len(behaviors)
    if n_behaviors == 0:
        print("No behaviors to visualize.")
        return {}

    # Adjust perplexity if needed (must be less than n_samples)
    adjusted_perplexity = min(perplexity, n_behaviors - 1)
    if adjusted_perplexity < 1:
        adjusted_perplexity = 1

    # Run t-SNE
    print(f"Running t-SNE with perplexity={adjusted_perplexity}...")
    tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # Plot t-SNE
    colors = plt.cm.tab10(np.linspace(0, 1, n_behaviors))
    ax1.scatter(coords[:, 0], coords[:, 1], c=colors, s=150, alpha=0.8, edgecolors='black')

    # Add labels
    for i, behavior in enumerate(behaviors):
        name = behavior.get("name", f"Behavior {i}")
        wrapped_name = "\n".join(textwrap.wrap(name, width=15))
        ax1.annotate(
            wrapped_name,
            (coords[i, 0], coords[i, 1]),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=8,
            alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
        )

    ax1.set_title("Behavior Embeddings (t-SNE)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax1.grid(True, alpha=0.3)

    # Plot semantic similarity matrix
    im = ax2.imshow(semantic_similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_title("Semantic Similarity (Cosine)", fontsize=14, fontweight='bold')

    # Add labels to similarity matrix
    names = [b.get("name", f"B{i}")[:15] for i, b in enumerate(behaviors)]
    ax2.set_xticks(range(n_behaviors))
    ax2.set_yticks(range(n_behaviors))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(names, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Cosine Similarity')

    # Add similarity values to cells
    for i in range(n_behaviors):
        for j in range(n_behaviors):
            text_color = 'white' if semantic_similarity_matrix[i, j] < 0.5 else 'black'
            ax2.text(j, i, f'{semantic_similarity_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=7, color=text_color)

    # Plot lexical similarity matrix
    im_lex = ax3.imshow(lexical_similarity_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax3.set_title("Lexical Similarity (ROUGE-L)", fontsize=14, fontweight="bold")
    ax3.set_xticks(range(n_behaviors))
    ax3.set_yticks(range(n_behaviors))
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax3.set_yticklabels(names, fontsize=8)

    cbar_lex = plt.colorbar(im_lex, ax=ax3, shrink=0.8)
    cbar_lex.set_label("ROUGE-L Similarity")

    for i in range(n_behaviors):
        for j in range(n_behaviors):
            text_color = "white" if lexical_similarity_matrix[i, j] < 0.5 else "black"
            ax3.text(j, i, f"{lexical_similarity_matrix[i, j]:.2f}",
                     ha="center", va="center", fontsize=7, color=text_color)

    # Suptitle with env, strategy, and diversity scores
    title_parts = ["Behavior Embedding Analysis"]
    if env_name:
        title_parts.append(f"Env: {env_name}")
    if strategy:
        title_parts.append(f"Strategy: {strategy}")
    title_parts.append(f"Semantic Div: {semantic_diversity:.2f}")
    title_parts.append(f"Lexical Div: {lexical_diversity:.2f}")
    fig.suptitle("  |  ".join(title_parts), fontsize=11, y=0.98)

    plt.tight_layout()
    plt.savefig(f"{output_path}/behaviors_tsne.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved t-SNE visualization to {output_path}/behaviors_tsne.png")

    return {
        "embeddings": embeddings,
        "semantic_similarity_matrix": semantic_similarity_matrix,
        "lexical_similarity_matrix": lexical_similarity_matrix,
        "semantic_diversity": semantic_diversity,
        "lexical_diversity": lexical_diversity,
    }

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Analyze behavior diversity from pipeline output.")
    parser.add_argument("run_dir", type=str, help="Path to pipeline output directory (e.g., results/football__stratified__gpt-5-mini)")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory for results. Defaults to run_dir/plots")
    parser.add_argument("--env_name", type=str, default=None, help="Environment name (auto-detected from run_dir if not provided)")
    parser.add_argument("--strategy", type=str, default=None, help="Strategy name (auto-detected from run_dir if not provided)")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-8B", help="Sentence transformer model name")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-5-mini", help="LLM model for quality evaluation (litellm format)")
    parser.add_argument("--perplexity", type=float, default=5.0, help="t-SNE perplexity parameter")
    args = parser.parse_args()

    # Load behaviors from run directory
    behaviors_path = os.path.join(args.run_dir, "behaviors.json")
    if not os.path.exists(behaviors_path):
        raise FileNotFoundError(f"behaviors.json not found in {args.run_dir}")

    with open(behaviors_path, "r") as f:
        behaviors = json.load(f)

    behaviors_str = _behaviors_to_texts(behaviors)

    # Auto-detect env_name and strategy from directory name (format: env__strategy__model)
    run_dir_name = os.path.basename(os.path.normpath(args.run_dir))
    parts = run_dir_name.split("__")
    env_name = args.env_name or (parts[0] if len(parts) >= 1 else None)
    strategy = args.strategy or (parts[1] if len(parts) >= 2 else None)

    metadata = {"env_name": env_name, "strategy": strategy}

    # Default output path
    output_path = args.output_path or os.path.join(args.run_dir, "plots")
    os.makedirs(output_path, exist_ok=True)

    env_module = importlib.import_module(f"popgen.envs.{env_name}")
    env_description = env_module.ENVIRONMENT_DESCRIPTION

    embeddings_path = os.path.join(output_path, "embeddings.npy")
    # Pass save_path=None to avoid directory issues; we'll compute embeddings for plotting below
    diversity_evaluator = DiversityEvaluator(
        embedding_model=args.embedding_model,
        save_path=None, 
        num_workers=1,
    )
    diversity_results = diversity_evaluator.evaluate([{"texts": behaviors_str}], metadata=metadata)
    diversity_output_path = os.path.join(output_path, "diversity_scores.json")
    diversity_evaluator.save_results(diversity_results, diversity_output_path)
    print(f"Saved diversity scores to {diversity_output_path}")

    semantic_diversity = float(diversity_results.overall_metrics["semantic_diversity"]["mean"])
    lexical_diversity = float(diversity_results.overall_metrics["lexical_diversity"]["mean"])
    instance_metrics = diversity_results.instance_metrics[0]
    if instance_metrics.get("semantic_similarity") is None:
        semantic_similarity_matrix = np.eye(len(behaviors_str))
    else:
        semantic_similarity_matrix = np.array(instance_metrics["semantic_similarity"])
    if instance_metrics.get("lexical_similarity") is None:
        lexical_similarity_matrix = np.eye(len(behaviors_str))
    else:
        lexical_similarity_matrix = np.array(instance_metrics["lexical_similarity"])

    # Load embeddings
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
    else:
        embeddings = diversity_evaluator.model.encode(behaviors_str, convert_to_numpy=True) if behaviors_str else np.array([])

    results = plot_behavior_tsne(
        behaviors,
        output_path=output_path,
        embeddings=embeddings,
        semantic_similarity_matrix=semantic_similarity_matrix,
        lexical_similarity_matrix=lexical_similarity_matrix,
        semantic_diversity=semantic_diversity,
        lexical_diversity=lexical_diversity,
        perplexity=args.perplexity,
        env_name=env_name,
        strategy=strategy,
    )

    quality_evaluator = BehaviorQualityEvaluator(judge_model=args.judge_model)
    quality_inputs = [
        {"behavior": behavior_text, "environment_description": env_description}
        for behavior_text in _behaviors_to_texts(behaviors)
    ]
    quality_results = quality_evaluator.evaluate(
        quality_inputs,
        metadata=metadata,
    )
    quality_output_path = os.path.join(output_path, "quality_scores.json")
    quality_evaluator.save_results(quality_results, quality_output_path)
    print(f"Saved quality scores to {quality_output_path}")

    print(f"\nDiversity Summary:")
    print(f"  Mean: {(semantic_diversity + lexical_diversity) / 2:.3f}")
    print(f"  Semantic: {results['semantic_diversity']:.3f}")
    print(f"  Lexical:  {results['lexical_diversity']:.3f}")
    print(f"\nQuality Summary:")
    quality_overall = quality_results.overall_metrics
    print(f"  Mean: {(quality_overall['relevance']['mean'] + quality_overall['realism']['mean'] + quality_overall['specificity']['mean']) / 3:.3f}")
    print(f"  Relevance: {quality_overall['relevance']['mean']:.3f}")
    print(f"  Realism:   {quality_overall['realism']['mean']:.3f}")
    print(f"  Specificity: {quality_overall['specificity']['mean']:.3f}")
