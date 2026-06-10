#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Submit one SLURM job per (dataset, seed) for synthetic-stream evaluation.
#
# Edit the CONFIG block below to control:
#   - the dataset list
#   - the seed list
#   - the ensemble-size sweep
#   - the decision heuristics (level 1 / level 2 / windows)
#   - the TP-matching tolerance
#
# Usage:
#     ./submit_synthetic.sh
# ----------------------------------------------------------------------------
set -euo pipefail

# ============================== CONFIG ======================================

DATASETS=(
    "SineClustersPre()"
    "WaveformPre()"
)

SEEDS=(1 2 3 4 5 6 7 8 9 10)

SIZES="2,4,8,16,32,64,128"

DETECTOR_DECISION_CRITERIA="majority"   # any | all | majority
ENSEMBLE_DECISION_CRITERIA="majority"   # any | all | majority
DECISION_WINDOW=5
SUPPRESSION_WINDOW=10
TOLERANCE=100
DETECTOR_BASE_SEED=1337
RECENT_SAMPLES_SIZE=100                  # empty = leave detectors as configured

CONFIG_DIR="scalability_configs"
OUT_DIR="synthetic_results"

# ============================ end CONFIG ====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/run_synthetic.sbatch"

mkdir -p "${OUT_DIR}"

for dataset_expr in "${DATASETS[@]}"; do
    dataset_label="${dataset_expr%%(*}"
    echo "=== Dataset: ${dataset_label} ==="

    for seed in "${SEEDS[@]}"; do
        cfg="${CONFIG_DIR}/seed${seed}/seed${seed}_128.yaml"
        if [[ ! -f "${cfg}" ]]; then
            echo "  [skip] missing config: ${cfg}" >&2
            continue
        fi

        echo "  Submitting ${dataset_label} seed=${seed}"
        sbatch \
            --job-name="Synth_${dataset_label}_s${seed}" \
            --export=ALL,\
DATASET="${dataset_expr}",\
SEED="${seed}",\
CONFIG="${cfg}",\
SIZES="${SIZES}",\
TOLERANCE="${TOLERANCE}",\
SUPPRESSION_WINDOW="${SUPPRESSION_WINDOW}",\
DECISION_WINDOW="${DECISION_WINDOW}",\
DETECTOR_DECISION_CRITERIA="${DETECTOR_DECISION_CRITERIA}",\
ENSEMBLE_DECISION_CRITERIA="${ENSEMBLE_DECISION_CRITERIA}",\
DETECTOR_BASE_SEED="${DETECTOR_BASE_SEED}",\
RECENT_SAMPLES_SIZE="${RECENT_SAMPLES_SIZE}",\
OUT_DIR="${OUT_DIR}" \
            "${SBATCH_TEMPLATE}"
    done
done

echo "All synthetic-stream jobs submitted."
