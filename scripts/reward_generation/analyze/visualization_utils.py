import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


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
    """Visualize behaviors using t-SNE and similarity matrix, with diversity scores."""
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
    ax1.scatter(coords[:, 0], coords[:, 1], c=colors, s=150, alpha=0.8, edgecolors="black")

    # Add labels
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

    ax1.set_title("Analyzed Behavior Embeddings (t-SNE)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax1.grid(True, alpha=0.3)

    # Plot semantic similarity matrix
    im = ax2.imshow(semantic_similarity_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax2.set_title("Semantic Similarity (Cosine)", fontsize=14, fontweight="bold")

    # Add labels to similarity matrix
    # names = [b.get("name", f"B{i}")[:15] for i, b in enumerate(behaviors)]
    # Use index as labels to keep it clean
    names = [f"B{i}" for i in range(n_behaviors)]

    ax2.set_xticks(range(n_behaviors))
    ax2.set_yticks(range(n_behaviors))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(names, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Cosine Similarity")

    # Add similarity values to cells
    for i in range(n_behaviors):
        for j in range(n_behaviors):
            text_color = "white" if semantic_similarity_matrix[i, j] < 0.5 else "black"
            ax2.text(j, i, f"{semantic_similarity_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

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
            ax3.text(j, i, f"{lexical_similarity_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    # Suptitle with env, strategy, and diversity scores
    title_parts = ["Trajectory Diversity Analysis"]
    if env_name:
        title_parts.append(f"Env: {env_name}")
    if strategy:
        title_parts.append(f"Strategy: {strategy}")
    title_parts.append(f"Semantic Div: {semantic_diversity:.2f}")
    title_parts.append(f"Lexical Div: {lexical_diversity:.2f}")
    fig.suptitle("  |  ".join(title_parts), fontsize=11, y=0.98)

    plt.tight_layout()
    plt.savefig(f"{output_path}/trajectory_diversity_tsne.png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved t-SNE visualization to {output_path}/trajectory_diversity_tsne.png")

    return {
        "embeddings": embeddings,
        "semantic_similarity_matrix": semantic_similarity_matrix,
        "lexical_similarity_matrix": lexical_similarity_matrix,
        "semantic_diversity": semantic_diversity,
        "lexical_diversity": lexical_diversity,
    }


def plot_similarity_comparison(
    behavior_names: list[str],
    output_path: str,
    embeddings_desired: np.ndarray,
    embeddings_analyzed: np.ndarray,
    semantic_scores: Any,  # Can be List or np.ndarray
    lexical_scores: list[float],
    semantic_mean: float,
    lexical_mean: float,
    env_name: str = None,
    strategy: str = None,
) -> None:
    """Visualize similarity comparison using t-SNE and similarity matrix."""
    n_behaviors = len(behavior_names)
    if n_behaviors == 0:
        return

    # 1. t-SNE Calculation
    # Concatenate embeddings to project into shared space
    all_embeddings = np.concatenate([embeddings_desired, embeddings_analyzed], axis=0)

    # Perplexity must be less than n_samples
    n_samples = all_embeddings.shape[0]
    perplexity = min(5, n_samples - 1) if n_samples > 1 else 1

    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(all_embeddings)

    coords_desired = coords[:n_behaviors]
    coords_analyzed = coords[n_behaviors:]

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # --- Plot 1: t-SNE ---
    colors = plt.cm.tab10(np.linspace(0, 1, n_behaviors))

    # Plot connections
    for i in range(n_behaviors):
        ax1.plot([coords_desired[i, 0], coords_analyzed[i, 0]], [coords_desired[i, 1], coords_analyzed[i, 1]], color="gray", alpha=0.3, linestyle="--")

    # Plot points
    ax1.scatter(coords_desired[:, 0], coords_desired[:, 1], c=colors, marker="o", s=100, label="Desired", edgecolors="black")
    ax1.scatter(coords_analyzed[:, 0], coords_analyzed[:, 1], c=colors, marker="^", s=100, label="Analyzed", edgecolors="black")

    # Annotate
    for i, name in enumerate(behavior_names):
        # Label near desired
        ax1.annotate(f"D-{i}", (coords_desired[i, 0], coords_desired[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        # Label near analyzed
        ax1.annotate(f"A-{i}", (coords_analyzed[i, 0], coords_analyzed[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax1.set_title("Behavior Alignment (t-SNE)\nCircles=Desired, Triangles=Analyzed", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    # ax1.legend()

    # --- Plot 2: Semantic Similarity Matrix ---
    # Handle if semantic_scores is already a matrix (from updated SimilarityEvaluator)
    # or a list (old version). If list, we can't plot full matrix, so fallback to bar.

    is_matrix = hasattr(semantic_scores, "ndim") and semantic_scores.ndim == 2 or isinstance(semantic_scores, list) and len(semantic_scores) > 0 and isinstance(semantic_scores[0], list)

    if is_matrix:
        matrix = np.array(semantic_scores)
        im = ax2.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax2.set_title(f"Semantic Similarity Matrix (Mean: {semantic_mean:.2f})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Analyzed Behavior ID")
        ax2.set_ylabel("Desired Behavior ID")

        # Ticks
        ax2.set_xticks(np.arange(n_behaviors))
        ax2.set_yticks(np.arange(n_behaviors))

        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label("Cosine Similarity")

        # Annotate cells
        for i in range(n_behaviors):
            for j in range(n_behaviors):
                text_color = "white" if matrix[i, j] < 0.5 else "black"
                ax2.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    else:
        # Fallback to Bar Chart
        y_pos = np.arange(n_behaviors)
        ax2.barh(y_pos, semantic_scores, color="skyblue", edgecolor="black")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(behavior_names, fontsize=8)
        ax2.set_xlabel("Cosine Similarity")
        ax2.set_title(f"Semantic Similarity (Mean: {semantic_mean:.2f})", fontsize=12, fontweight="bold")
        ax2.set_xlim(0, 1.0)
        ax2.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(semantic_scores):
            ax2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    # --- Plot 3: Lexical Similarity Matrix ---
    is_matrix_lex = hasattr(lexical_scores, "ndim") and lexical_scores.ndim == 2 or isinstance(lexical_scores, list) and len(lexical_scores) > 0 and isinstance(lexical_scores[0], list)

    if is_matrix_lex:
        matrix = np.array(lexical_scores)
        im = ax3.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax3.set_title(f"Lexical Similarity Matrix (Mean: {lexical_mean:.2f})", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Analyzed Behavior ID")
        ax3.set_ylabel("Desired Behavior ID")

        # Ticks
        ax3.set_xticks(np.arange(n_behaviors))
        ax3.set_yticks(np.arange(n_behaviors))

        # Colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label("ROUGE-L Similarity")

        # Annotate cells
        for i in range(n_behaviors):
            for j in range(n_behaviors):
                text_color = "white" if matrix[i, j] < 0.5 else "black"
                ax3.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)
    else:
        y_pos = np.arange(n_behaviors)
        ax3.barh(y_pos, lexical_scores, color="lightgreen", edgecolor="black")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(behavior_names, fontsize=8)  # Optionally hide labels here if redundant
        ax3.set_xlabel("ROUGE-L F1")
        ax3.set_title(f"Lexical Similarity (Mean: {lexical_mean:.2f})", fontsize=12, fontweight="bold")
        ax3.set_xlim(0, 1.0)
        ax3.grid(True, axis="x", alpha=0.3)

        for i, v in enumerate(lexical_scores):
            ax3.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    # Global Title
    title_parts = ["Trajectory Similarity Analysis"]
    if env_name:
        title_parts.append(f"Env: {env_name}")
    if strategy:
        title_parts.append(f"Strategy: {strategy}")
    fig.suptitle("  |  ".join(title_parts), fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig(f"{output_path}/similarity_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved similarity plot to {output_path}/similarity_comparison.png")
