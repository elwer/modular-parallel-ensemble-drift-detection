#!/bin/bash
# Submit one SLURM job per drift detector per dataset for independent optimization.

DETECTORS=("BNDM" "CSDDM" "D3" "IBDD" "OCDD" "SPLL" "UDetect")
#DATASETS=("RialtoBridgeTimelapse" "PokerHand")
DATASETS=("Electricity" "GasSensor" "ForestCovertype")
N_SUCCESSFUL=${1:-1000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/optimize_single_dd.sbatch"

for DATASET in "${DATASETS[@]}"; do
    echo "=== Dataset: ${DATASET} ==="

    for DETECTOR in "${DETECTORS[@]}"; do
        echo "  Submitting job for ${DETECTOR} on ${DATASET} (target: ${N_SUCCESSFUL} successful runs)..."
        sbatch --job-name="DD_${DETECTOR}_${DATASET}" \
               --export=ALL,DETECTOR="${DETECTOR}",N_TRIALS="${N_SUCCESSFUL}",DATASET="${DATASET}" \
               "${SBATCH_TEMPLATE}"
    done
done

echo "All jobs submitted."
