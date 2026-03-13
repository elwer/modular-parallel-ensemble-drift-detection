#!/bin/bash
# Submit one SLURM job per detector per fold for cross-dataset optimization.
# Leave-one-out: 8 detectors x 5 folds = 40 jobs.

DETECTORS=("BNDM" "CSDDM" "D3" "IBDD" "OCDD" "SPLL" "UDetect" "MOPEDDS")
N_TRIALS=${1:-1000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/optimize_cross_dataset.sbatch"

COUNT=0
for DETECTOR in "${DETECTORS[@]}"; do
    for FOLD in $(seq 0 4); do
        echo "Submitting ${DETECTOR} fold ${FOLD} (${N_TRIALS} trials)..."
        sbatch --job-name="CrossDD_${DETECTOR}_f${FOLD}" \
               --export=ALL,DETECTOR="${DETECTOR}",N_TRIALS="${N_TRIALS}",FOLD="${FOLD}" \
               "${SBATCH_TEMPLATE}"
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted ${COUNT} jobs."
