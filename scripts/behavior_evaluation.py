"""Behavior diversity and quality analysis."""

from __future__ import annotations

import importlib
import json
import os
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from popgen.population.evaluation import (
    BehaviorQualityEvaluator,
    DiversityEvaluator,
    RougeLDiversity,
)
from popgen.utils.config import LitellmConfig


def _behaviors_to_texts(behaviors: list[dict[str, Any]]) -> list[str]:
    """Convert behaviors to text representations."""
    return [f"{b.get('name', '')}: {b.get('description', '')}" for b in behaviors]


def plot_behavior_tsne(
    behaviors: list[dict[str, Any]],
    output_path: str,
    embeddings: np.ndarray,
    similarity_matrix: np.ndarray,
    rouge_l_distance_matrix: np.ndarray | None = None,
    avg_pairwise_distance: float = 0.0,
    avg_rouge_l_distance: float = 0.0,
    perplexity: float = 5.0,
    env_name: str = None,
    strategy: str = None,
) -> dict[str, Any]:
    """Visualize behaviors using t-SNE and similarity matrices.

    Args:
        behaviors: List of dicts with keys `name` and `description`.
        output_path: Directory path to save outputs.
        embeddings: Embedding matrix for behaviors.
        similarity_matrix: Pairwise cosine similarity matrix.
        rouge_l_distance_matrix: Pairwise ROUGE-L distance matrix (optional).
        avg_pairwise_distance: Average pairwise embedding distance.
        avg_rouge_l_distance: Average pairwise ROUGE-L distance.
        perplexity: t-SNE perplexity parameter.
        env_name: Optional environment name to display.
        strategy: Optional strategy name to display.

    Returns:
        Dict containing embeddings, similarity matrix, and diversity scores.
    """
    n_behaviors = len(behaviors)
    if n_behaviors == 0:
        print("No behaviors to visualize.")
        return {}

    adjusted_perplexity = min(perplexity, max(1, n_behaviors - 1))

    print(f"Running t-SNE with perplexity={adjusted_perplexity}...")
    tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    has_rouge = rouge_l_distance_matrix is not None and rouge_l_distance_matrix.size > 0
    n_cols = 3 if has_rouge else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7))

    # t-SNE scatter
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_behaviors))
    ax1.scatter(coords[:, 0], coords[:, 1], c=colors, s=150, alpha=0.8, edgecolors="black")
    for i, behavior in enumerate(behaviors):
        name = behavior.get("name", f"Behavior {i}")
        wrapped_name = "\n".join(textwrap.wrap(name, width=15))
        ax1.annotate(
            wrapped_name,
            (coords[i, 0], coords[i, 1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
        )
    ax1.set_title("Behavior Embeddings (t-SNE)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax1.grid(True, alpha=0.3)

    # Cosine similarity heatmap
    ax2 = axes[1]
    names = [b.get("name", f"B{i}")[:15] for i, b in enumerate(behaviors)]
    im = ax2.imshow(similarity_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax2.set_title("Cosine Similarity", fontsize=14, fontweight="bold")
    ax2.set_xticks(range(n_behaviors))
    ax2.set_yticks(range(n_behaviors))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(names, fontsize=8)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Cosine Similarity")
    for i in range(n_behaviors):
        for j in range(n_behaviors):
            text_color = "white" if similarity_matrix[i, j] < 0.5 else "black"
            ax2.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    # ROUGE-L distance heatmap (if available)
    if has_rouge:
        ax3 = axes[2]
        im_lex = ax3.imshow(rouge_l_distance_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax3.set_title("ROUGE-L Distance", fontsize=14, fontweight="bold")
        ax3.set_xticks(range(n_behaviors))
        ax3.set_yticks(range(n_behaviors))
        ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax3.set_yticklabels(names, fontsize=8)
        cbar_lex = plt.colorbar(im_lex, ax=ax3, shrink=0.8)
        cbar_lex.set_label("ROUGE-L Distance")
        for i in range(n_behaviors):
            for j in range(n_behaviors):
                text_color = "white" if rouge_l_distance_matrix[i, j] > 0.5 else "black"
                ax3.text(j, i, f"{rouge_l_distance_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    # Title
    title_parts = ["Behavior Embedding Analysis"]
    if env_name:
        title_parts.append(f"Env: {env_name}")
    if strategy:
        title_parts.append(f"Strategy: {strategy}")
    title_parts.append(f"Avg Pairwise Dist: {avg_pairwise_distance:.3f}")
    if has_rouge:
        title_parts.append(f"Avg ROUGE-L Dist: {avg_rouge_l_distance:.3f}")
    fig.suptitle("  |  ".join(title_parts), fontsize=11, y=0.98)

    plt.tight_layout()
    plt.savefig(f"{output_path}/behaviors_tsne.png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved t-SNE visualization to {output_path}/behaviors_tsne.png")

    return {
        "embeddings": embeddings,
        "similarity_matrix": similarity_matrix,
        "avg_pairwise_distance": avg_pairwise_distance,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze behavior diversity from pipeline output.")
    parser.add_argument("run_dir", type=str, help="Path to pipeline output directory")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory for results. Defaults to run_dir/plots")
    parser.add_argument("--env_name", type=str, default=None, help="Environment name (auto-detected from run_dir if not provided)")
    parser.add_argument("--strategy", type=str, default=None, help="Strategy name (auto-detected from run_dir if not provided)")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-8B", help="Sentence transformer model name")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-5-mini", help="LLM model for quality evaluation (litellm format)")
    parser.add_argument("--perplexity", type=float, default=5.0, help="t-SNE perplexity parameter")
    args = parser.parse_args()

    # Load behaviors
    behaviors_path = os.path.join(args.run_dir, "behaviors.json")
    if not os.path.exists(behaviors_path):
        raise FileNotFoundError(f"behaviors.json not found in {args.run_dir}")

    with open(behaviors_path) as f:
        behaviors = json.load(f)

    behaviors_str = _behaviors_to_texts(behaviors)

    # Auto-detect env_name and strategy from directory name (format: env__strategy__model)
    run_dir_name = os.path.basename(os.path.normpath(args.run_dir))
    parts = run_dir_name.split("__")
    env_name = args.env_name or (parts[0] if len(parts) >= 1 else None)
    strategy = args.strategy or (parts[1] if len(parts) >= 2 else None)

    metadata = {"env_name": env_name, "strategy": strategy}

    output_path = args.output_path or os.path.join(args.run_dir, "plots")
    os.makedirs(output_path, exist_ok=True)

    env_module = importlib.import_module(f"popgen.envs.{env_name}")
    env_description = env_module.ENVIRONMENT_DESCRIPTION

    # --- Embedding diversity ---
    diversity_evaluator = DiversityEvaluator(
        embedding_model=args.embedding_model,
        save_path=output_path,
        num_workers=1,
    )
    diversity_results = diversity_evaluator.evaluate([{"texts": behaviors_str}], metadata=metadata)
    diversity_output_path = os.path.join(output_path, "diversity_scores.json")
    diversity_evaluator.save_results(diversity_results, diversity_output_path)
    print(f"Saved diversity scores to {diversity_output_path}")

    instance_metrics = diversity_results.instance_metrics[0]
    overall = diversity_results.overall_metrics

    similarity_matrix = np.array(instance_metrics.get("similarity_matrix", np.eye(len(behaviors_str))))
    avg_pairwise_distance = float(overall["average_pairwise_distance"]["mean"])

    # Load embeddings (saved by evaluator)
    embeddings_path = os.path.join(output_path, "embeddings_0.npy")
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
    else:
        embeddings = diversity_evaluator.encoder.batch_encoder(behaviors_str)
        embeddings = np.asarray(embeddings, dtype=float)

    # --- ROUGE-L diversity ---
    rouge_evaluator = RougeLDiversity(num_workers=1)
    rouge_results = rouge_evaluator.evaluate([{"texts": behaviors_str}], metadata=metadata)
    rouge_instance = rouge_results.instance_metrics[0]
    rouge_l_distance_matrix = np.array(rouge_instance.get("distance_matrix", np.array([])))
    avg_rouge_l_distance = float(rouge_results.overall_metrics["pairwise_rouge_l_distance"]["mean"])

    rouge_output_path = os.path.join(output_path, "rouge_l_scores.json")
    rouge_evaluator.save_results(rouge_results, rouge_output_path)
    print(f"Saved ROUGE-L scores to {rouge_output_path}")

    # --- Plot ---
    results = plot_behavior_tsne(
        behaviors,
        output_path=output_path,
        embeddings=embeddings,
        similarity_matrix=similarity_matrix,
        rouge_l_distance_matrix=rouge_l_distance_matrix,
        avg_pairwise_distance=avg_pairwise_distance,
        avg_rouge_l_distance=avg_rouge_l_distance,
        perplexity=args.perplexity,
        env_name=env_name,
        strategy=strategy,
    )

    # --- Quality evaluation ---
    judge_config = LitellmConfig(model_name=args.judge_model)
    quality_evaluator = BehaviorQualityEvaluator(model_config=judge_config)
    quality_inputs = [
        {"behavior": behavior_text, "environment_description": env_description}
        for behavior_text in _behaviors_to_texts(behaviors)
    ]
    quality_results = quality_evaluator.evaluate(quality_inputs, metadata=metadata)
    quality_output_path = os.path.join(output_path, "quality_scores.json")
    quality_evaluator.save_results(quality_results, quality_output_path)
    print(f"Saved quality scores to {quality_output_path}")

    # --- Summary ---
    print("\nDiversity Summary:")
    print(f"  Avg Pairwise Distance: {avg_pairwise_distance:.3f}")
    print(f"  Min Pairwise Distance: {float(overall['minimum_pairwise_distance']['mean']):.3f}")
    print(f"  Coverage:              {float(overall['coverage']['mean']):.3f}")
    print(f"  Dispersion:            {float(overall['dispersion']['mean']):.3f}")
    print(f"  Avg ROUGE-L Distance:  {avg_rouge_l_distance:.3f}")

    print("\nQuality Summary:")
    quality_overall = quality_results.overall_metrics
    print(f"  Relevance:   {quality_overall['relevance']['mean']:.3f}")
    print(f"  Realism:     {quality_overall['realism']['mean']:.3f}")
    print(f"  Specificity: {quality_overall['specificity']['mean']:.3f}")
