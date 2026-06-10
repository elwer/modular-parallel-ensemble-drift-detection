#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Synthetic-stream ensemble scalability experiments (local runner).
#
# Iterates over (DATASET x SEED) and invokes main_synthetic.py once per pair
# with --sizes covering the full ensemble-size sweep. Each run consumes the
# corresponding seed{N}_128.yaml pool (main_synthetic.py slices the prefix
# for each smaller size internally).
#
# Edit the variables in the CONFIG block below and run:
#     ./run_synthetic.sh
#
# Outputs are written under ${OUT_DIR}/<dataset>/seed<N>.log
# ----------------------------------------------------------------------------
set -euo pipefail

# ============================== CONFIG ======================================

DATASETS=(
    "SineClustersPre()"
    "WaveformPre()"
)

SEEDS=(1 2 3 4 5 6 7 8 9 10)

SIZES="2,4,8,16,32,64,128"

# Decision heuristics (forwarded to main_synthetic.py)
DETECTOR_DECISION_CRITERIA="majority"   # any | all | majority
ENSEMBLE_DECISION_CRITERIA="majority"   # any | all | majority
DECISION_WINDOW=1                       # int >= 1
SUPPRESSION_WINDOW=1000                 # int >= 0 (samples)

# TP-matching tolerance (samples after a known drift point)
TOLERANCE=1000

# Optional: override recent_samples_size for every detector ("" = leave as-is).
RECENT_SAMPLES_SIZE=""

# Base seed forwarded to detectors (combined with slot index in main_synthetic).
DETECTOR_BASE_SEED=1337

CONFIG_DIR="scalability_configs"
OUT_DIR="synthetic_results"

# ============================ end CONFIG ====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${OUT_DIR}"

# Optional venv setup if setup.sh exists; safe to skip otherwise.
if [[ -f setup.sh && -z "${SKIP_SETUP:-}" ]]; then
    # shellcheck disable=SC1091
    source setup.sh || true
fi

extra_args=()
if [[ -n "${RECENT_SAMPLES_SIZE}" ]]; then
    extra_args+=(--recent-samples-size "${RECENT_SAMPLES_SIZE}")
fi

for dataset_expr in "${DATASETS[@]}"; do
    # Strip "()" and any args for the directory label.
    dataset_label="${dataset_expr%%(*}"
    dataset_dir="${OUT_DIR}/${dataset_label}"
    mkdir -p "${dataset_dir}"

    for seed in "${SEEDS[@]}"; do
        cfg="${CONFIG_DIR}/seed${seed}/seed${seed}_128.yaml"
        if [[ ! -f "${cfg}" ]]; then
            echo "[skip] missing config: ${cfg}" >&2
            continue
        fi

        log="${dataset_dir}/seed${seed}.log"
        echo "=== ${dataset_label} | seed=${seed} -> ${log}"

        python main_synthetic.py \
            "${dataset_expr}" "${cfg}" \
            --sizes "${SIZES}" \
            --tolerance "${TOLERANCE}" \
            --suppression-window "${SUPPRESSION_WINDOW}" \
            --decision-window "${DECISION_WINDOW}" \
            --detector-decision-criteria "${DETECTOR_DECISION_CRITERIA}" \
            --ensemble-decision-criteria "${ENSEMBLE_DECISION_CRITERIA}" \
            --seed "${DETECTOR_BASE_SEED}" \
            "${extra_args[@]}" \
            2>&1 | tee "${log}"
    done
done

echo "All runs finished. Logs under ${OUT_DIR}/"
