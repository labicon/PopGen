"""Behavior visualization using graphviz."""

from __future__ import annotations
from typing import List, Dict, Any
from graphviz import Digraph
import json
import html
import textwrap

def draw_behaviors(behaviors: List[Dict[str, Any]], output_path: str = "output/behaviors_viz") -> None:
    """Render a visualization of behaviors using graphviz.

    Args:
        behaviors: List of dicts with keys `name` (str) and `description` (str).
        output_path: Path to save the rendered image (without extension).
    """
    dot = Digraph(comment="Behaviors")
    dot.attr(rankdir="LR")  # Left to right layout
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', margin='0.2')
    
    root_id = "root"
    dot.node(root_id, "Generated Behaviors", shape="doubleoctagon", style="filled", fillcolor="lightgrey", fontsize="16")

    for i, behavior in enumerate(behaviors):
        name = html.escape(behavior.get("name", "Unnamed"))
        description = behavior.get("description", "")
        
        # Wrap description
        wrapped_lines = textwrap.wrap(description, width=40)
        escaped_lines = [html.escape(line) for line in wrapped_lines]
        formatted_description = "<BR/>".join(escaped_lines)
        
        node_id = f"behavior_{i}"
        # Increased font size and added bold to name
        label = f"<{name}<BR/><BR/><FONT POINT-SIZE='12' COLOR='#555555'>{formatted_description}</FONT>>"
        
        dot.node(node_id, label)
        dot.edge(root_id, node_id)

    dot.render(output_path, format="png", cleanup=True)
    print(f"Saved visualization to {output_path}.png")

if __name__ == "__main__":
    # Load your behaviors
    with open("output/verbalized_sampling/behaviors_run_3.json", "r") as f:
        behaviors = json.load(f)

    # Visualize
    draw_behaviors(behaviors, "output/my_behaviors_viz")