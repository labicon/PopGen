#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present
ENV_FILE="${SCRIPT_DIR}/../../.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

# ---------------------------------------------------------------------------
# Usage:
#   ./run_experiments.sh                          # full pipeline
#   ./run_experiments.sh env=pp_stag  # full pipeline, only pp_stag env
#   ./run_experiments.sh --cooperator-only <dir>  # skip self-play, train
#                                                 # cooperator against an
#                                                 # existing population dir
# ---------------------------------------------------------------------------
COOPERATOR_ONLY=false
SELF_PLAY_DIR=""
ENVS=(overcooked)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cooperator-only)
            COOPERATOR_ONLY=true
            SELF_PLAY_DIR="${2:?'--cooperator-only requires a directory argument'}"
            shift 2
            ;;
        env=*)
            ENVS=("${1#env=}")
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done
STRATEGIES=(self_play)
N_SEEDS_LIST=(4)
N_RUNS=1
N_EVAL_ROLLOUTS="${N_EVAL_ROLLOUTS:-50}"

failed=0

if [[ "${COOPERATOR_ONLY}" == "true" ]]; then
    # Derive env from the directory path (…/results/<env>/self_play/…)
    env="$(basename "$(dirname "$(dirname "${SELF_PLAY_DIR}")")")"
    run_name="cooperator_$(date +%Y-%m-%d_%H-%M-%S)"
    echo "=== cooperator-only | self_play_dir=${SELF_PLAY_DIR}  env=${env} ==="
    (
        cd "${SCRIPT_DIR}"
        python main.py \
            "env=${env}" \
            "pipeline.run_self_play=false" \
            "cooperator.self_play_results_dir=${SELF_PLAY_DIR}" \
            "cooperator.run_name=${run_name}"
    ) || {
        echo "ERROR: cooperator training failed."
        exit 1
    }
    echo "Done."
    exit 0
fi

for strategy in "${STRATEGIES[@]}"; do
    for env in "${ENVS[@]}"; do
        for n_seeds in "${N_SEEDS_LIST[@]}"; do
            echo "=== strategy=${strategy}  env=${env}  n_seeds=${n_seeds} ==="
            for i in $(seq 1 ${N_RUNS}); do
                run_name="${strategy}_n${n_seeds}_run_${i}_$(date +%Y-%m-%d_%H-%M-%S)"
                run_dir="${SCRIPT_DIR}/results/${env}/self_play/${run_name}"
                population_seed=$(((RANDOM << 16) ^ RANDOM))
                worker_override_args=()
                if [[ "${env}" == "smacv2" ]]; then
                    worker_override_args=("self_play.n_workers=1")
                fi

                echo "--- n_seeds=${n_seeds} run ${i}/${N_RUNS}: training (run_name=${run_name}, population_seed=${population_seed}, n_workers=${worker_override_args[0]:-default}) ---"
                (
                    cd "${SCRIPT_DIR}"
                    python main.py \
                        "env=${env}" \
                        "self_play.n_seeds=${n_seeds}" \
                        "self_play.population_seed=${population_seed}" \
                        "${worker_override_args[@]}" \
                        "self_play.run_name=${run_name}" \
                        "cooperator.run_name=${run_name}" \
                        "pipeline.run_evaluation=true"
                ) || {
                    echo "WARNING: training failed for n_seeds=${n_seeds} run ${i}/${N_RUNS} (env=${env}), continuing..."
                    ((failed++)) || true
                    continue
                }
            done
        done
    done
done

if [[ ${failed} -gt 0 ]]; then
    echo "Done. ${failed} run(s) failed."
else
    echo "Done. All runs succeeded."
fi
