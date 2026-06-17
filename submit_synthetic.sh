#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Submit one SLURM job per ablation point for synthetic-stream evaluation.
#
# Ablation grid: (dataset x seed x detector_criterion x ensemble_criterion
#                 x decision_window x suppression_window)
#
# Edit the CONFIG block below to control:
#   - the dataset list
#   - the seed list
#   - the ensemble-size sweep
#   - the ablation grids (criteria and window sizes)
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

# --- Ablation grids ---
DETECTOR_DECISION_CRITERIA_LIST=(any all majority)   # level 1
ENSEMBLE_DECISION_CRITERIA_LIST=(any all majority)   # level 2
DECISION_WINDOWS=(10 20 30 40 50)                    # level-1 window, steps of 10
SUPPRESSION_WINDOWS=(10 20 30 40 50)                 # FP/TP collapse window, steps of 10

TOLERANCE=100
DETECTOR_BASE_SEED=1337
RECENT_SAMPLES_SIZE=100                  # empty = leave detectors as configured

CONFIG_DIR="scalability_configs"
OUT_DIR="synthetic_results"

# ============================ end CONFIG ====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/run_synthetic.sbatch"

mkdir -p "${OUT_DIR}"
n_jobs=0

for dataset_expr in "${DATASETS[@]}"; do
    dataset_label="${dataset_expr%%(*}"
    echo "=== Dataset: ${dataset_label} ==="

    for seed in "${SEEDS[@]}"; do
        cfg="${CONFIG_DIR}/seed${seed}/seed${seed}_128.yaml"
        if [[ ! -f "${cfg}" ]]; then
            echo "  [skip] missing config: ${cfg}" >&2
            continue
        fi

        for det_crit in "${DETECTOR_DECISION_CRITERIA_LIST[@]}"; do
            for ens_crit in "${ENSEMBLE_DECISION_CRITERIA_LIST[@]}"; do
                for dw in "${DECISION_WINDOWS[@]}"; do
                    for sw in "${SUPPRESSION_WINDOWS[@]}"; do
                        echo "  Submitting ${dataset_label} seed=${seed}" \
                             "det=${det_crit} ens=${ens_crit} dw=${dw} sw=${sw}"
                        # NOTE: variables are passed via the environment
                        # (--export=ALL). Listing them inside --export=...
                        # would break SIZES, because sbatch splits the
                        # --export argument on commas.
                        DATASET="${dataset_expr}" \
                        SEED="${seed}" \
                        CONFIG="${cfg}" \
                        SIZES="${SIZES}" \
                        TOLERANCE="${TOLERANCE}" \
                        SUPPRESSION_WINDOW="${sw}" \
                        DECISION_WINDOW="${dw}" \
                        DETECTOR_DECISION_CRITERIA="${det_crit}" \
                        ENSEMBLE_DECISION_CRITERIA="${ens_crit}" \
                        DETECTOR_BASE_SEED="${DETECTOR_BASE_SEED}" \
                        RECENT_SAMPLES_SIZE="${RECENT_SAMPLES_SIZE}" \
                        OUT_DIR="${OUT_DIR}" \
                        sbatch \
                            --job-name="Synth_${dataset_label}_s${seed}_${det_crit}_${ens_crit}_dw${dw}_sw${sw}" \
                            --export=ALL \
                            "${SBATCH_TEMPLATE}"
                        n_jobs=$((n_jobs + 1))
                        sleep 1
                    done
                done
            done
        done
    done
done

echo "All synthetic-stream jobs submitted (${n_jobs} jobs)."
