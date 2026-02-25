#!/bin/bash
# Run the full population generation pipeline:
#   1. generate_behaviors.py  – generate behavior descriptions
#   2. generate_agents.py     – generate & evaluate agents for each behavior
#   3. gather_results.py      – collect metrics into a CSV
#
# Usage (from scripts/population/):
#   ./run_pipeline.sh [HYDRA_OVERRIDES...]
#
# Examples:
#   ./run_pipeline.sh
#   ./run_pipeline.sh env=lbf generation=policy_code generation.model_name=openrouter/google/gemini-3-flash-preview n_behaviors=5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "$(date +'%H:%M:%S') | $*"; }

# Derive the Hydra output_dir from overrides + config defaults.
# Mirrors: ./results/${env}/${generation.type}/${basename:${generation.model_name}}
compute_output_dir() {
    python3 - "$@" <<'PYEOF'
import sys, re

# Defaults matching cfg/config.yaml + cfg/generation/policy_code.yaml
env = "lbf"
gen_type = "policy_code"
model_name = "openrouter/google/gemini-3-flash-preview"

# Map generation config-group name → type (matches the yaml files)
gen_type_map = {
    "policy_code": "policy_code",
    "rl_reward": "rl_reward",
    "llm_agent": "llm_agent",
}

for arg in sys.argv[1:]:
    if "=" not in arg:
        continue
    key, val = arg.split("=", 1)
    if key == "env":
        env = val
    elif key == "generation":
        gen_type = gen_type_map.get(val, val)
    elif key == "generation.type":
        gen_type = val
    elif key == "generation.model_name":
        model_name = val

model_basename = model_name.split("/")[-1]
print(f"results/{env}/{gen_type}/{model_basename}")
PYEOF
}

# ---------------------------------------------------------------------------
# Step 1: generate behaviors
# ---------------------------------------------------------------------------

log "=== Step 1: Generating behaviors ==="
behav_log=$(python generate_behaviors.py "$@" 2>&1 | tee /dev/stderr)

# Extract path from either:
#   "Saved N behavior(s) to PATH"
#   "Loading cached behaviors from PATH"
behavior_path=$(printf '%s\n' "$behav_log" \
    | grep -E "(behavior\(s\) to|behaviors from) " \
    | awk '{print $NF}' \
    | tail -1)

if [[ -z "$behavior_path" ]]; then
    log "ERROR: could not extract behavior file path from generate_behaviors.py output" >&2
    exit 1
fi
log "Behavior file: $behavior_path"

# ---------------------------------------------------------------------------
# Step 2: generate agents
# ---------------------------------------------------------------------------

log ""
log "=== Step 2: Generating agents ==="
python generate_agents.py "behavior_path=$behavior_path" "$@"

# ---------------------------------------------------------------------------
# Step 3: gather results
# ---------------------------------------------------------------------------

log ""
log "=== Step 3: Gathering results ==="
output_dir=$(compute_output_dir "$@")

if [[ ! -d "$output_dir" ]]; then
    log "WARNING: output_dir '$output_dir' not found, skipping gather_results" >&2
else
    csv_path="$output_dir/results.csv"
    python gather_results.py "$output_dir" -o "$csv_path"
    log "Results saved to $csv_path"
fi

log ""
log "=== Pipeline complete ==="
