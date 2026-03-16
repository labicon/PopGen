#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENVS=(lbf)
STRATEGIES=(verbalized_sampling)
N_RUNS=5
N_EVAL_ROLLOUTS="${N_EVAL_ROLLOUTS:-50}"

failed=0

for strategy in "${STRATEGIES[@]}"; do
    for env in "${ENVS[@]}"; do
        echo "=== strategy=${strategy}  env=${env} ==="
        for i in $(seq 1 ${N_RUNS}); do
            run_name="${strategy}_run_${i}_$(date +%Y-%m-%d_%H-%M-%S)"
            run_dir="${SCRIPT_DIR}/results/${env}/self_play/${run_name}"

            echo "--- run ${i}/${N_RUNS}: training (run_name=${run_name}) ---"
            (
                cd "${SCRIPT_DIR}"
                python main.py \
                    "env=${env}" \
                    "self_play.run_name=${run_name}" \
                    "cooperator.run_name=${run_name}" \
                    "pipeline.run_evaluation=false"
            ) || {
                echo "WARNING: training failed for run ${i}/${N_RUNS} (env=${env}), continuing..."
                ((failed++)) || true
                continue
            }

            echo "--- run ${i}/${N_RUNS}: evaluation (run_dir=${run_dir}) ---"
            (
                cd "${SCRIPT_DIR}"
                python evaluation.py \
                    "env=${env}" \
                    "evaluation.run_dir=${run_dir}" \
                    "evaluation.n_eval_rollouts=${N_EVAL_ROLLOUTS}"
            ) || {
                echo "WARNING: evaluation failed for run ${i}/${N_RUNS} (env=${env}), continuing..."
                ((failed++)) || true
            }
        done
    done
done

if [[ ${failed} -gt 0 ]]; then
    echo "Done. ${failed} run(s) failed."
else
    echo "Done. All runs succeeded."
fi
