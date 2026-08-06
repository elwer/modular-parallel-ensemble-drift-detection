#!/bin/bash
# SLURM batch script for default configs experiment.
#
# Required env vars (set via sbatch --export):
#   DRIFT_FREQS          Comma-separated drift frequencies
#   STREAM_LENGTH        Stream length
#   N_DEPLOY_TRIALS      Deployment optimization trials
#   N_FOLDS              Number of CV folds
#   N_CPUS               Number of CPUs
#   GENERALIST_TIMEOUT   Per-trial timeout for grid evaluation (seconds)
#   OUTPUT_DIR           Output directory

#SBATCH --job-name=DefExp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

DRIFT_FREQS="${DRIFT_FREQS:-200,400,500,750,1000,1250,1500,2000}"
STREAM_LENGTH="${STREAM_LENGTH:-5000}"
N_DEPLOY_TRIALS="${N_DEPLOY_TRIALS:-50}"
N_FOLDS="${N_FOLDS:-10}"
N_CPUS="${N_CPUS:-16}"
GENERALIST_TIMEOUT="${GENERALIST_TIMEOUT:-600}"
OUTPUT_DIR="${OUTPUT_DIR:-results_defaults_experiment_hpc}"

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/experiment.log"

python -u run_defaults_experiment.py \
    --drift-freqs "${DRIFT_FREQS}" \
    --stream-length "${STREAM_LENGTH}" \
    --n-deploy-trials "${N_DEPLOY_TRIALS}" \
    --n-folds "${N_FOLDS}" \
    --n-cpus "${N_CPUS}" \
    --generalist-timeout "${GENERALIST_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG}"
