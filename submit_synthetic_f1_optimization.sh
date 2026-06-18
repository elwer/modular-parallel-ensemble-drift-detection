#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Submit one 24-hour F1-optimization SLURM job per (ensemble size, optimizer)
# pair for a single synthetic dataset.
#
# Two optimizers are launched per size:
#   - Optuna (TPE)   -> optimization/synthetic_f1_optimize_optuna.py
#   - BoTorch (Ax)   -> optimization/synthetic_f1_optimize_botorch.py
#
# Both jointly optimize:
#   - MOPEDDS-level params
#   - Per-slot detector type (which DDs are in the ensemble)
#   - Detector hyperparameters
# (See each script's docstring for the precise search-space differences.)
#
# Run once per dataset, e.g.:
#     ./submit_synthetic_f1_optimization.sh "SineClustersPre()"
#     ./submit_synthetic_f1_optimization.sh "WaveformPre()"
#
# Optional 2nd arg overrides the ensemble-size list (space-separated).
# Optional 3rd arg overrides the optimizer set: "optuna botorch", "optuna",
# or "botorch" (default: both).
# ----------------------------------------------------------------------------
set -euo pipefail

DATASET="${1:-SineClustersPre()}"
SIZES_STR="${2:-1 2 4 8 16 32 64 128}"
METHODS_STR="${3:-optuna botorch}"
read -r -a SIZES   <<< "${SIZES_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

TIMEOUT=86400        # 24 hours in seconds
TOLERANCE=100
SEED=1337
OPTUNA_OUT_DIR="synthetic_optuna_results"
BOTORCH_OUT_DIR="synthetic_botorch_results"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTUNA_SBATCH="${SCRIPT_DIR}/optimize_synthetic_f1.sbatch"
BOTORCH_SBATCH="${SCRIPT_DIR}/optimize_synthetic_f1_botorch.sbatch"

dataset_label="${DATASET%%(*}"
mkdir -p "${OPTUNA_OUT_DIR}/${dataset_label}"
mkdir -p "${BOTORCH_OUT_DIR}/${dataset_label}"

n_jobs=0
for size in "${SIZES[@]}"; do
    for method in "${METHODS[@]}"; do
        case "${method}" in
            optuna)
                out_dir="${OPTUNA_OUT_DIR}"
                sbatch_template="${OPTUNA_SBATCH}"
                job_name="SynthF1opt_${dataset_label}_N${size}"
                ;;
            botorch)
                out_dir="${BOTORCH_OUT_DIR}"
                sbatch_template="${BOTORCH_SBATCH}"
                job_name="SynthF1bo_${dataset_label}_N${size}"
                ;;
            *)
                echo "  [skip] unknown method '${method}' (expected: optuna|botorch)" >&2
                continue
                ;;
        esac

        echo "Submitting ${dataset_label} N=${size} method=${method} (24h F1 optimization)"
        DATASET="${DATASET}" \
        SIZE="${size}" \
        TIMEOUT="${TIMEOUT}" \
        TOLERANCE="${TOLERANCE}" \
        SEED="${SEED}" \
        OUT_DIR="${out_dir}" \
        sbatch \
            --job-name="${job_name}" \
            --export=ALL \
            "${sbatch_template}"
        n_jobs=$((n_jobs + 1))
        sleep 1
    done
done

echo "Submitted ${n_jobs} F1 optimization jobs for ${dataset_label}."
