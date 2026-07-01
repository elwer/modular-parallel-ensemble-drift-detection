#!/bin/bash
# SLURM template for greedy-ensemble-30min-steps run.
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
#   OPTUNA_STORAGE        Optuna storage URL (e.g., sqlite:///optuna.db)
#   MAX_N                 Maximum ensemble size (default: 16)
#   N_WORKERS             Number of parallel Optuna workers (default: 64)
#   STEP_TIMEOUT_HOURS    Timeout per greedy step (default: 0.5 = 30min)
#   SEED                  Base detector seed
#   OUT_DIR               Output directory

#SBATCH --job-name=Greedy30min
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
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
OUTPUT_CSV="${OUT_DIR}/${TAG}/greedy_30min_${TAG}_S${N_STREAMS}.csv"
LOG="${OUT_DIR}/${TAG}/greedy_30min_${TAG}_S${N_STREAMS}.log"

EXTRA_ARGS=()
if [[ -n "${GENERATORS:-}" ]]; then
    EXTRA_ARGS+=(--generators "${GENERATORS}")
elif [[ -n "${GENERATOR:-}" ]]; then
    EXTRA_ARGS+=(--generator "${GENERATOR}")
else
    echo "ERROR: must set GENERATORS." >&2
    exit 2
fi
if [[ -n "${STREAM_SEEDS:-}" ]]; then
    EXTRA_ARGS+=(--stream-seeds "${STREAM_SEEDS}")
fi
if [[ -n "${TOLERANCES:-}" ]]; then
    EXTRA_ARGS+=(--tolerances "${TOLERANCES}")
fi
if [[ -n "${MAX_N:-}" ]]; then
    EXTRA_ARGS+=(--max-n "${MAX_N}")
fi
if [[ -n "${STEP_TIMEOUT_HOURS:-}" ]]; then
    EXTRA_ARGS+=(--step-timeout-hours "${STEP_TIMEOUT_HOURS}")
fi
if [[ -n "${N_TRIALS_STEP1:-}" ]]; then
    EXTRA_ARGS+=(--n-trials-step1 "${N_TRIALS_STEP1}")
fi
if [[ -n "${N_TRIALS_STEP2_PLUS:-}" ]]; then
    EXTRA_ARGS+=(--n-trials-step2-plus "${N_TRIALS_STEP2_PLUS}")
fi

python optimization/greedy_ensemble_30min_steps.py \
    --optuna-storage "${OPTUNA_STORAGE}" \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --eval-stream-indices "${EVAL_STREAM_INDICES}" \
    --max-n "${MAX_N:-16}" \
    --n-workers "${N_WORKERS:-64}" \
    --step-timeout-hours "${STEP_TIMEOUT_HOURS:-0.5}" \
    --seed "${SEED:-1337}" \
    --output-csv "${OUTPUT_CSV}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"
