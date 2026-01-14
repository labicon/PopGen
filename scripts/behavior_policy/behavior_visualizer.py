"""Behavior visualization using matplotlib."""

from __future__ import annotations
from typing import List, Dict, Any
import json
import math
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_behaviors(
    behaviors: List[Dict[str, Any]],
    output_path: str = "behaviors_viz",
    env_name: str = None,
    strategy: str = None,
) -> None:
    """Render a visualization of behaviors using matplotlib.

    Args:
        behaviors: List of dicts with keys `name` (str) and `description` (str).
        output_path: Path to save the rendered image (without extension).
        env_name: Optional environment name to display below title.
        strategy: Optional strategy name to display below title.
    """
    n_behaviors = len(behaviors)
    if n_behaviors == 0:
        print("No behaviors to visualize.")
        return

    # Layout configuration - aim for square grid
    cols = math.ceil(math.sqrt(n_behaviors))
    rows = math.ceil(n_behaviors / cols)

    fig_width = cols * 4
    fig_height = rows * 3 + 1.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, cols)
    ax.set_ylim(-0.2, rows + 0.6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(cols / 2, rows + 0.35, "Generated Behaviors",
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', edgecolor='black'))

    # Subtitle (env_name and strategy)
    subtitle_parts = []
    if env_name:
        subtitle_parts.append(f"Environment: {env_name}")
    if strategy:
        subtitle_parts.append(f"Strategy: {strategy}")
    if subtitle_parts:
        subtitle = "  |  ".join(subtitle_parts)
        ax.text(cols / 2, rows + 0.05, subtitle,
                ha='center', va='center', fontsize=10, color='#555555')

    # Draw behavior cards
    card_width = 0.85
    card_height = 0.85

    for i, behavior in enumerate(behaviors):
        row = rows - 1 - (i // cols)
        col = i % cols

        x = col + 0.5
        y = row + 0.4

        name = behavior.get("name", "Unnamed")
        description = behavior.get("description", "")

        # Wrap text
        wrapped_name = "\n".join(textwrap.wrap(name, width=20))
        wrapped_desc = "\n".join(textwrap.wrap(description, width=25))

        # Truncate description if too long
        desc_lines = wrapped_desc.split("\n")
        if len(desc_lines) > 6:
            desc_lines = desc_lines[:6]
            desc_lines[-1] = desc_lines[-1][:20] + "..."
        wrapped_desc = "\n".join(desc_lines)

        # Draw card background
        card = mpatches.FancyBboxPatch(
            (x - card_width/2, y - card_height/2),
            card_width, card_height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor='white',
            edgecolor='#333333',
            linewidth=1.5
        )
        ax.add_patch(card)

        # Draw name
        ax.text(x, y + card_height/2 - 0.15, wrapped_name,
                ha='center', va='top', fontsize=10, fontweight='bold',
                wrap=True)

        # Draw description
        ax.text(x, y + card_height/2 - 0.3, wrapped_desc,
                ha='center', va='top', fontsize=7, color='#555555',
                wrap=True)

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved visualization to {output_path}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize behaviors from a JSON file.")
    parser.add_argument("--behaviors_path", type=str, required=True, help="Path to behaviors JSON file")
    parser.add_argument("--output_path", type=str, default="behaviors_viz", help="Output path for visualization (without extension)")
    parser.add_argument("--env_name", type=str, default=None, help="Environment name to display")
    parser.add_argument("--strategy", type=str, default=None, help="Strategy name to display")
    args = parser.parse_args()

    with open(args.behaviors_path, "r") as f:
        behaviors = json.load(f)

    draw_behaviors(behaviors, args.output_path, env_name=args.env_name, strategy=args.strategy)
