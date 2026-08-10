#!/bin/bash
# SLURM batch script for ensemble vs generalist comparison.
#
# Required env vars (set via sbatch --export):
#   DRIFT_FREQS          Comma-separated drift frequencies
#   STREAM_LENGTH        Stream length
#   N_BUDGET             Expert optimization trials per stream
#   N_DEPLOY_TRIALS      Deployment optimization trials
#   N_FOLDS              Number of CV folds
#   N_CPUS               Number of CPUs
#   EXPERT_TIMEOUT       Per-trial timeout for experts (seconds)
#   GENERALIST_TIMEOUT   Per-trial timeout for generalists (seconds)
#   OUTPUT_DIR           Output directory

#SBATCH --job-name=EnsVsGen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8192

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/eliasw/modular-parallel-ensemble-drift-detection}"
source setup.sh

# Single-threaded BLAS/numpy — parallelism is via ProcessPoolExecutor
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHON_GIL=0

DRIFT_FREQS="${DRIFT_FREQS:-200,500,1000}"
STREAM_LENGTH="${STREAM_LENGTH:-2000}"
N_BUDGET="${N_BUDGET:-60}"
N_BUDGET_MAX="${N_BUDGET_MAX:-60}"
N_BUDGET_STEP="${N_BUDGET_STEP:-15}"
ENSEMBLE_F1_THRESHOLD="${ENSEMBLE_F1_THRESHOLD:-0.0}"
EXPERT_MIN_VAL_F1="${EXPERT_MIN_VAL_F1:-0.3}"
N_DEPLOY_TRIALS="${N_DEPLOY_TRIALS:-30}"
N_FOLDS="${N_FOLDS:-10}"
N_CPUS="${N_CPUS:-32}"
EXPERT_TIMEOUT="${EXPERT_TIMEOUT:-120}"
GENERALIST_TIMEOUT="${GENERALIST_TIMEOUT:-600}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ensemble_vs_generalist_hpc}"

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/experiment.log"

python -u run_ensemble_vs_generalist.py \
    --drift-freqs "${DRIFT_FREQS}" \
    --stream-length "${STREAM_LENGTH}" \
    --n-budget "${N_BUDGET}" \
    --n-budget-max "${N_BUDGET_MAX}" \
    --n-budget-step "${N_BUDGET_STEP}" \
    --ensemble-f1-threshold "${ENSEMBLE_F1_THRESHOLD}" \
    --expert-min-val-f1 "${EXPERT_MIN_VAL_F1}" \
    --n-deploy-trials "${N_DEPLOY_TRIALS}" \
    --n-folds "${N_FOLDS}" \
    --n-cpus "${N_CPUS}" \
    --expert-timeout "${EXPERT_TIMEOUT}" \
    --generalist-timeout "${GENERALIST_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG}"
