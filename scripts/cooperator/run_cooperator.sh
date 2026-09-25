#!/usr/bin/env bash
# Run cooperator training + evaluation on an existing population.
#
# Usage (from scripts/cooperator/):
#   ./run_cooperator.sh env=lbf population_dir=/path/to/population
#   ./run_cooperator.sh env=football population_dir=/path/to/population n_eval_rollouts=20

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYGLET_HEADLESS=true
export PYGLET_SHADOW_WINDOW=0

log() { echo "$(date +'%H:%M:%S') | $*"; }

# ---------------------------------------------------------------------------
# Configuration – override via key=value arguments
# ---------------------------------------------------------------------------

env="point_mass_rendezvous"
population_dir=""
n_eval_rollouts=50

for arg in "$@"; do
    case "$arg" in
        env=*)             env="${arg#env=}" ;;
        population_dir=*)  population_dir="${arg#population_dir=}" ;;
        n_eval_rollouts=*) n_eval_rollouts="${arg#n_eval_rollouts=}" ;;
        *) echo "WARNING: unknown argument: ${arg}" >&2 ;;
    esac
done

if [[ -z "${population_dir}" ]]; then
    echo "ERROR: population_dir is required (e.g. population_dir=/path/to/population)" >&2
    exit 1
fi

if [[ ! -d "${population_dir}" ]]; then
    echo "ERROR: population_dir not found: ${population_dir}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Cooperator training
# ---------------------------------------------------------------------------

run_tag="$(date +%Y%m%d_%H%M%S)"
cooperator_log="${population_dir}/cooperator-${run_tag}.log"

log "=== Cooperator training ==="
cd "${script_dir}"
python generate_cooperator.py \
    "population.env=${env}" \
    "population.output_dir=${population_dir}" \
    "cooperator.run_name=${env}-cooperator-${run_tag}" \
    "cooperator.results_dir=${population_dir}" 2>&1 | tee "${cooperator_log}"

cooperator_dir=$(grep "COOPERATOR_OUTPUT_DIR:" "${cooperator_log}" | awk '{print $NF}' | tail -1 || true)

# ---------------------------------------------------------------------------
# Step 2: Evaluation
# ---------------------------------------------------------------------------

log ""
log "=== Evaluation ==="
if [[ -n "${cooperator_dir}" ]]; then
    python gather_results.py \
        --cooperator_dir "${cooperator_dir}" \
        --env "${env}" \
        --n_eval_rollouts "${n_eval_rollouts}" \
        --results_dir "${population_dir}"
    log "Results saved to ${population_dir}"
else
    log "WARNING: could not extract cooperator output dir; skipping evaluation" >&2
fi
