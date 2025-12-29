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
SIMPLESTRAT_BEHAVIOR_OUTPUT = "results/behavior/simple_strat/"
IN_CONTEXT_BEHAVIOR_OUTPUT = "results/behavior/in_context/"
POLICY_OUTPUT = "results/policy_code/"
ROLLOUT_OUTPUT = "results/rollout/"
EVALUATION_OUTPUT = "results/evaluation/"
TREE_IMAGE = "output/behavior_tree"  # graphviz will add extension

# Generation sizes
MAX_BEHAVIORS = 5

# Source file for environment rollout
ROLLOUT_SOURCE_FILE = "policy_rollout/envs/rollout_point_mass_rendezvous.py"
POLICY_FILE_PATH = "policy_rollout/envs/agent1_policy.py"

ENVIRONMENT_PROMPT_FILE = "prompt/point_mass_rendezvous.py"

