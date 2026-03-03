#!/bin/bash
# Submit one SLURM job per dataset for EWDD pre-selected ensemble optimization.
# Requires that single-DD optimization results exist in results/ for each dataset.

#DATASETS=("RialtoBridgeTimelapse" "PokerHand")
DATASETS=("Electricity" "GasSensor" "ForestCovertype")
N_TRIALS=${1:-500}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/optimize_ewdd_preselected.sbatch"

for DATASET in "${DATASETS[@]}"; do
    echo "Submitting EWDD PreSelected job for ${DATASET} (${N_TRIALS} trials, all Pareto candidates/detector)..."
    sbatch --job-name="DD_EWDD_PreSel_${DATASET}" \
           --export=ALL,N_TRIALS="${N_TRIALS}",DATASET="${DATASET}" \
           "${SBATCH_TEMPLATE}"
done

echo "All jobs submitted."
