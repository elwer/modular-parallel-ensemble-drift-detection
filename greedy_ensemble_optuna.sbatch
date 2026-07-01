#!/bin/bash
# SLURM template for greedy-ensemble-optuna run.
#
# Required env vars (set via sbatch --export):
#   GENERATORS            Comma-separated per-stream list (length 1 or N_STREAMS)
#   STUDY_TAG             Output tag (subdir + CSV name)
#   N_STREAMS             Number of streams
#   BASE_STREAM_SEED      Base stream seed
#   DRIFT_FREQUENCIES     Comma-separated drift frequencies
#   STREAM_LENGTH         Stream length
#   EVAL_STREAM_INDICES   Comma-separated eval indices
#   TOLERANCES            Comma-separated tolerances (optional)
#   POOL_GLOB             Glob pattern for N=1 CSV pool
#   OPTUNA_STORAGE        Optuna storage URL (e.g., sqlite:///optuna.db)
#   MAX_N                 Maximum ensemble size
#   N_TRIALS              Number of Optuna trials per step (optional, default: unlimited)
#   N_WORKERS             Number of parallel Optuna workers
#   STEP_TIMEOUT_HOURS    Timeout per greedy step (hours)
#   GLOBALS_MODE          'best' or 'manual'
#   SEED                  Base detector seed
#   OUT_DIR               Output directory

#SBATCH --job-name=GreedyOptuna
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/data/horse/ws/s4122485-parallelDD/new/modular-parallel-ensemble-drift-detection}"
source setup.sh

# Optuna uses multiprocessing for parallel trials
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TAG="${STUDY_TAG}"
mkdir -p "${OUT_DIR}/${TAG}"
OUTPUT_CSV="${OUT_DIR}/${TAG}/greedy_optuna_${TAG}_S${N_STREAMS}.csv"
LOG="${OUT_DIR}/${TAG}/greedy_optuna_${TAG}_S${N_STREAMS}.log"

EXTRA_ARGS=()
if [[ -n "${GENERATORS:-}" ]]; then
    EXTRA_ARGS+=(--generators "${GENERATORS}")
elif [[ -n "${GENERATOR:-}" ]]; then
    EXTRA_ARGS+=(--generator "${GENERATOR}")
else
    echo "ERROR: must set GENERATORS." >&2
    exit 2
fi
if [[ -n "${STUDY_TAG:-}" ]]; then
    EXTRA_ARGS+=(--study-tag "${STUDY_TAG}")
fi
if [[ -n "${EVAL_STREAM_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--eval-stream-indices "${EVAL_STREAM_INDICES}")
fi
if [[ -n "${TOLERANCES:-}" ]]; then
    EXTRA_ARGS+=(--tolerances "${TOLERANCES}")
fi
if [[ -n "${STREAM_SEEDS:-}" ]]; then
    EXTRA_ARGS+=(--stream-seeds "${STREAM_SEEDS}")
fi
if [[ -n "${N_TRIALS:-}" ]]; then
    EXTRA_ARGS+=(--n-trials "${N_TRIALS}")
fi

python optimization/greedy_ensemble_optuna.py \
    --pool-glob "${POOL_GLOB}" \
    --optuna-storage "${OPTUNA_STORAGE}" \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --max-n "${MAX_N}" \
    --n-workers "${N_WORKERS}" \
    --step-timeout-hours "${STEP_TIMEOUT_HOURS}" \
    --globals "${GLOBALS_MODE}" \
    --seed "${SEED}" \
    --output-csv "${OUTPUT_CSV}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"
