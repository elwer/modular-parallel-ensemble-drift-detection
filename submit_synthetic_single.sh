#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Single-member counterpart of submit_synthetic.sh.
#
# Runs the same synthetic-stream ablation study, but with an ensemble of size
# 1 (just the first detector from each seed's pool of 128). All 10 seeds are
# still iterated. The full ablation grid (det_crit x ens_crit x dw x sw) is
# still swept; with a single member the criteria collapse to the same outcome
# in practice, but we keep the grid for symmetry with the multi-member runs.
#
# Usage:
#     ./submit_synthetic_single.sh
# ----------------------------------------------------------------------------
set -euo pipefail

# ============================== CONFIG ======================================

DATASETS=(
    "SineClustersPre()"
    "WaveformPre()"
)

SEEDS=(1 2 3 4 5 6 7 8 9 10)

SIZES="1"

# --- Ablation grids ---
DETECTOR_DECISION_CRITERIA_LIST=(any all majority)   # level 1
ENSEMBLE_DECISION_CRITERIA_LIST=(any all majority)   # level 2
DECISION_WINDOWS=(10 20 30 40 50)                    # level-1 window, steps of 10
SUPPRESSION_WINDOWS=(10 20 30 40 50)                 # FP/TP collapse window, steps of 10

TOLERANCE=100
DETECTOR_BASE_SEED=1337
RECENT_SAMPLES_SIZE=100                  # empty = leave detectors as configured

CONFIG_DIR="scalability_configs"
OUT_DIR="synthetic_results_single"

# ============================ end CONFIG ====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/run_synthetic.sbatch"

mkdir -p "${OUT_DIR}"
n_jobs=0

for dataset_expr in "${DATASETS[@]}"; do
    dataset_label="${dataset_expr%%(*}"
    echo "=== Dataset: ${dataset_label} ==="

    for seed in "${SEEDS[@]}"; do
        # Reuse the 128-detector pool config; main_synthetic.py will only
        # instantiate the first detector when SIZES="1".
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
                             "size=1 det=${det_crit} ens=${ens_crit} dw=${dw} sw=${sw}"
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
                            --job-name="SynthSingle_${dataset_label}_s${seed}_${det_crit}_${ens_crit}_dw${dw}_sw${sw}" \
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

echo "All single-member synthetic-stream jobs submitted (${n_jobs} jobs)."
