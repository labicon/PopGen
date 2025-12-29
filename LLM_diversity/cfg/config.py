"""Configuration constants for the agentic framework.

All values are simple top-level constants as requested. Environment variables
should be used for secrets; `OPENAI_API_KEY` is left as `None` here to signal
that behavior.
"""

# LLM model selection
LLM_MODEL = "gpt-5"

# If you want to hardcode a key you can, but by default we use env vars.
OPENAI_API_KEY = "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"

# Output file paths
SIMPLESTRAT_BEHAVIOR_OUTPUT = "output/simple_strat/behaviors.json"
VERBALIZED_SAMPLING_OUTPUT = "output/verbalized_sampling/behaviors.json"
IN_CONTEXT_BEHAVIOR_OUTPUT = "output/in_context/behaviors.json"
CURRICULUM_OUTPUT = "output/curriculum.json"
NEW_BEHAVIOR_TREE_OUTPUT = "output/behavior_tree_generation/behavior_trees.json"
BEHAVIOR_TREE_IN_CONTEXT_OUTPUT = "output/behavior_tree_generation/in_context.json"
BEHAVIOR_TREE_VERBALIZED_OUTPUT = "output/behavior_tree_generation/verbalized.json"
BEHAVIOR_TREE_STRATIFIED_OUTPUT = "output/behavior_tree_generation/stratified.json"
DIVERSITY_OUTPUT = "output/diversity.json"
TREE_IMAGE = "output/behavior_tree"  # graphviz will add extension

# Generation sizes
MAX_BEHAVIORS = 10
MAX_DIVERSE_BEHAVIORS = 10

