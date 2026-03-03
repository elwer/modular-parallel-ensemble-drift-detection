#!/bin/bash
# Submit runtime experiments: 10 repetitions × 3 core counts × 2 detector/config pairs = 60 jobs
#
# After all jobs finish, run:
#   python compute_mean_runtimes.py

REPETITIONS=10
CORE_COUNTS="1 2 4"

# (detector, config_id) pairs to benchmark
CONFIGS=(
    "spll 1"
    "csddm 31"
)

for entry in "${CONFIGS[@]}"; do
    read -r DETECTOR CONFIG_ID <<< "$entry"
    for CORES in ${CORE_COUNTS}; do
        for RUN_ID in $(seq 0 $((REPETITIONS - 1))); do
            OUTPUT_DIR="runtime_experiments/${DETECTOR}_${CONFIG_ID}/${CORES}_cores"
            mkdir -p "${OUTPUT_DIR}"
            echo "Submitting ${DETECTOR} config ${CONFIG_ID}, ${CORES} core(s), run ${RUN_ID}"
            sbatch \
                --job-name="RT_${DETECTOR}_c${CONFIG_ID}_${CORES}c_r${RUN_ID}" \
                --cpus-per-task=${CORES} \
                --output="${OUTPUT_DIR}/slurm_run${RUN_ID}_%j.out" \
                --export=ALL,DETECTOR=${DETECTOR},CONFIG_ID=${CONFIG_ID},CORES=${CORES},RUN_ID=${RUN_ID} \
                run_single_config.sbatch
        done
    done
done

echo ""
echo "Submitted $((${#CONFIGS[@]} * 3 * REPETITIONS)) jobs total."
echo "After all jobs finish, run:  python compute_mean_runtimes.py"
