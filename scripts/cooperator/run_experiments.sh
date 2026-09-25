#!/usr/bin/env bash
# Run one full cooperator experiment:
#   1. generate_behaviors.py  – generate a fresh behavior set
#   2. generate_agents.py     – generate & evaluate agents for each behavior
#   3. generate_cooperator.py – train cooperator on the population
#   4. gather_results.py      – evaluate cooperator and collect metrics
#
# Usage (from scripts/cooperator/):
#   ./run_experiments.sh
#   ./run_experiments.sh env=football n_behaviors=8
#   ./run_experiments.sh env=lbf behavior=in_context n_behaviors=12 generation.model_name=openrouter/google/gemini-3-flash-preview
#   ./run_experiments.sh env=lbf results_root=results_population_sweep
#   ./run_experiments.sh env=lbf results_dir=/absolute/path/to/save

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
population_scripts_dir="${repo_root}/scripts/population"
population_config_dir="${script_dir}/population_cfg"

export PYGLET_HEADLESS=true
export PYGLET_SHADOW_WINDOW=0

log() { echo "$(date +'%H:%M:%S') | $*"; }

# ---------------------------------------------------------------------------
# Derive run-specific paths from overrides + config defaults.
# Each invocation gets a unique run_tag so behaviors are always regenerated.
# Mirrors the compute_output_dir pattern from scripts/population/run_pipeline.sh.
# ---------------------------------------------------------------------------

run_tag="$(date +%Y%m%d_%H%M%S)"

compute_run_vars() {
    python3 - "$@" <<'PYEOF'
import sys

env = "lbf"
gen_type = "policy_code"
model_name = "openrouter/google/gemini-3-flash-preview"
strategy = "verbalized_sampling_multiturn"
n_behaviors = "4"
run_tag = ""
results_root = "results"
results_dir_override = ""

for arg in sys.argv[1:]:
    if "=" not in arg:
        continue
    key, val = arg.split("=", 1)
    if key == "env":
        env = val
    elif key == "generation":
        gen_type = val
    elif key == "generation.type":
        gen_type = val
    elif key == "generation.model_name":
        model_name = val
    elif key == "behavior":
        strategy = val
    elif key == "n_behaviors":
        n_behaviors = val
    elif key == "_run_tag":
        run_tag = val
    elif key == "results_root":
        results_root = val
    elif key == "results_dir":
        results_dir_override = val

model_basename = model_name.split("/")[-1]
results_dir = results_dir_override or f"{results_root}/{env}/{gen_type}/{model_basename}/{strategy}/n{n_behaviors}/{run_tag}"
print(f"run_env={env}")
print(f"run_results_dir={results_dir}")
PYEOF
}

eval "$(compute_run_vars "_run_tag=${run_tag}" "$@")"
# run_env and run_results_dir are now set

behavior_path_override=""
forwarded_args=()
for arg in "$@"; do
    case "${arg}" in
        results_root=*|results_dir=*)
            ;;
        behavior_path=*)
            behavior_path_override="${arg#behavior_path=}"
            ;;
        *)
            forwarded_args+=("${arg}")
            ;;
    esac
done

# Resolve results_dir: absolute if given explicitly, otherwise relative to script_dir
case "${run_results_dir}" in
    /*) results_dir="${run_results_dir}" ;;
    *)  results_dir="${script_dir}/${run_results_dir}" ;;
esac
behavior_path="${results_dir}/behaviors/${run_env}.json"
if [[ -n "${behavior_path_override}" ]]; then
    behavior_path="${behavior_path_override}"
fi

mkdir -p "${results_dir}/behaviors"

# ---------------------------------------------------------------------------
# Step 1: Generate behaviors (skipped if behavior_path_override is provided)
# ---------------------------------------------------------------------------

if [[ -z "${behavior_path_override}" ]]; then
    log "=== Step 1: Generating behaviors ==="
    cd "${script_dir}"
    python "${population_scripts_dir}/generate_behaviors.py" \
        --config-path "${population_config_dir}" \
        --config-name config \
        "behavior_path=${behavior_path}" \
        "output_dir=${results_dir}" \
        "${forwarded_args[@]}"

    if [[ ! -f "${behavior_path}" ]]; then
        log "ERROR: behavior file not found after generation: ${behavior_path}" >&2
        exit 1
    fi
    log "Behavior file: ${behavior_path}"
else
    log "=== Step 1: Skipped (using provided behavior_path=${behavior_path}) ==="
    if [[ ! -f "${behavior_path}" ]]; then
        log "ERROR: provided behavior_path does not exist: ${behavior_path}" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Generate agents
# ---------------------------------------------------------------------------

log ""
log "=== Step 2: Generating agents ==="
python "${population_scripts_dir}/generate_agents.py" \
    --config-path "${population_config_dir}" \
    --config-name config \
    "behavior_path=${behavior_path}" \
    "output_dir=${results_dir}" \
    "${forwarded_args[@]}"

# ---------------------------------------------------------------------------
# Step 3: Cooperator training
# ---------------------------------------------------------------------------

log ""
log "=== Step 3: Cooperator training ==="
cd "${script_dir}"
cooperator_log="${results_dir}/cooperator.log"
python generate_cooperator.py \
    "population.env=${run_env}" \
    "population.output_dir=${results_dir}" \
    "cooperator.run_name=${run_env}-cooperator-${run_tag}" \
    "cooperator.results_dir=${results_dir}" 2>&1 | tee "${cooperator_log}"

cooperator_dir=$(grep "COOPERATOR_OUTPUT_DIR:" "${cooperator_log}" | awk '{print $NF}' | tail -1 || true)

# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------

log ""
log "=== Step 4: Evaluation ==="
if [[ -n "${cooperator_dir}" ]]; then
    python gather_results.py \
        --cooperator_dir "${cooperator_dir}" \
        --env "${run_env}" \
        --results_dir "${results_dir}"
    log "Results saved to ${results_dir}"
else
    log "WARNING: could not extract cooperator output dir; skipping evaluation" >&2
fi

log ""
log "=== Done (run_tag=${run_tag}) ==="
