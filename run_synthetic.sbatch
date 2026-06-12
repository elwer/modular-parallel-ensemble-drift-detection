#!/bin/bash
# SLURM job template for one (dataset, seed) synthetic-stream evaluation.
# Submitted by submit_synthetic.sh.
#
# Required env vars (set via sbatch --export):
#   DATASET                       Dataset class expression, e.g. "SineClustersPre()"
#   SEED                          Seed of the scalability config to use
#   CONFIG                        Path to the seed{SEED}_128.yaml file
#   SIZES                         Comma-separated ensemble sizes
#   TOLERANCE                     TP-matching tolerance (samples)
#   SUPPRESSION_WINDOW            Suppression window (samples)
#   DECISION_WINDOW               Level-1 decision window (samples)
#   DETECTOR_DECISION_CRITERIA    any | all | majority
#   ENSEMBLE_DECISION_CRITERIA    any | all | majority
#   DETECTOR_BASE_SEED            Base seed forwarded to main_synthetic.py
#   OUT_DIR                       Root output directory
#   RECENT_SAMPLES_SIZE           Optional: empty string to leave unchanged.

#SBATCH --job-name=SyntheticDD
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

cd /data/quokka/ws/9/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection
source setup.sh

dataset_label="${DATASET%%(*}"
mkdir -p "${OUT_DIR}/${dataset_label}"
LOG="${OUT_DIR}/${dataset_label}/seed${SEED}_det-${DETECTOR_DECISION_CRITERIA}_ens-${ENSEMBLE_DECISION_CRITERIA}_dw${DECISION_WINDOW}_sw${SUPPRESSION_WINDOW}.log"

extra_args=()
if [[ -n "${RECENT_SAMPLES_SIZE:-}" ]]; then
    extra_args+=(--recent-samples-size "${RECENT_SAMPLES_SIZE}")
fi

python main_synthetic.py \
    "${DATASET}" "${CONFIG}" \
    --sizes "${SIZES}" \
    --tolerance "${TOLERANCE}" \
    --suppression-window "${SUPPRESSION_WINDOW}" \
    --decision-window "${DECISION_WINDOW}" \
    --detector-decision-criteria "${DETECTOR_DECISION_CRITERIA}" \
    --ensemble-decision-criteria "${ENSEMBLE_DECISION_CRITERIA}" \
    --seed "${DETECTOR_BASE_SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${LOG}"
