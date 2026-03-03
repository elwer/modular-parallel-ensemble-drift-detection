#!/bin/bash
# Submit one SLURM job per (detector, source_dataset, eval_dataset) triplet.
# ForestCovertype is excluded as an evaluation target.

DETECTORS=("BNDM" "CSDDM" "D3" "IBDD" "OCDD" "SPLL" "UDetect" "EWDD")
SOURCE_DATASETS=("Electricity" "GasSensor" "PokerHand" "RialtoBridgeTimelapse")
EVAL_DATASETS=("Electricity" "GasSensor" "PokerHand" "RialtoBridgeTimelapse")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/run_cross_dataset_evaluation.sbatch"

N_JOBS=0

for DETECTOR in "${DETECTORS[@]}"; do
    for SOURCE in "${SOURCE_DATASETS[@]}"; do
        for EVAL in "${EVAL_DATASETS[@]}"; do
            # Skip same-dataset pairs
            if [ "${SOURCE}" == "${EVAL}" ]; then
                continue
            fi

            echo "Submitting: ${DETECTOR} ${SOURCE} -> ${EVAL}"
            sbatch --job-name="CE_${DETECTOR}_${SOURCE}_${EVAL}" \
                   --export=ALL,DETECTOR="${DETECTOR}",SOURCE_DATASET="${SOURCE}",EVAL_DATASET="${EVAL}" \
                   "${SBATCH_TEMPLATE}"
            N_JOBS=$((N_JOBS + 1))
        done
    done
done

echo "Submitted ${N_JOBS} cross-dataset evaluation jobs."
