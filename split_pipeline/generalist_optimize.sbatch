#!/bin/bash
# SLURM template for generalist optimization + eval-on-holdout for a single DD type.
#
# Required env vars (set via sbatch --export):
#   OPTUNA_STORAGE        Optuna storage URL
#   DETECTOR_TYPE         Detector type (IBDD, OCDD, D3, SPLL, UDetect, CSDDM, BNDM)
#   N_STREAMS             Number of streams
#   BASE_STREAM_SEED      Base stream seed
#   DRIFT_FREQUENCIES     Comma-separated drift frequencies
#   STREAM_LENGTH         Stream length
#   EVAL_STREAM_INDICES   Comma-separated eval indices
#   GENERATORS            Comma-separated per-stream list
#   SEED                  Base detector seed
#   N_TRIALS              Number of Optuna trials (default: 800 total budget)
#   PER_TRIAL_TIMEOUT     Per-trial timeout in seconds
#   OUTPUT_DIR            Output directory

#SBATCH --job-name=GenOpt_${DETECTOR_TYPE}
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8192

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/eliasw/modular-parallel-ensemble-drift-detection}"
PIPELINE_DIR="${SLURM_SUBMIT_DIR:-/home/eliasw/modular-parallel-ensemble-drift-detection}/split_pipeline"
source setup.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///split_pipeline/split_pipeline_optuna.db}"
DETECTOR_TYPE="${DETECTOR_TYPE:-IBDD}"
N_STREAMS="${N_STREAMS:-10}"
BASE_STREAM_SEED="${BASE_STREAM_SEED:-42}"
DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES:-200,400,500,750,1000,1250,1500,2000,2500,3000}"
STREAM_LENGTH="${STREAM_LENGTH:-8000}"
EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES:-1,4,8}"
GENERATORS="${GENERATORS:-SineClusters}"
SEED="${SEED:-1337}"
N_TRIALS="${N_TRIALS:-800}"
PER_TRIAL_TIMEOUT="${PER_TRIAL_TIMEOUT:-1200}"
OUTPUT_DIR="${OUTPUT_DIR:-split_pipeline/results}"

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/generalist_optimize_${DETECTOR_TYPE}.log"

EXTRA_ARGS=()
if [[ -n "${GENERATORS:-}" ]]; then
    EXTRA_ARGS+=(--generators "${GENERATORS}")
elif [[ -n "${GENERATOR:-}" ]]; then
    EXTRA_ARGS+=(--generator "${GENERATOR}")
fi
if [[ -n "${STREAM_SEEDS:-}" ]]; then
    EXTRA_ARGS+=(--stream-seeds "${STREAM_SEEDS}")
fi
if [[ -n "${TOLERANCES:-}" ]]; then
    EXTRA_ARGS+=(--tolerances "${TOLERANCES}")
fi
if [[ -n "${PROFILES:-}" ]]; then
    EXTRA_ARGS+=(--profiles "${PROFILES}")
fi

python "${PIPELINE_DIR}/generalist_optimize.py" \
    --optuna-storage "${OPTUNA_STORAGE}" \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --eval-stream-indices "${EVAL_STREAM_INDICES}" \
    --detector-type "${DETECTOR_TYPE}" \
    --seed "${SEED}" \
    --n-trials "${N_TRIALS}" \
    --per-trial-timeout "${PER_TRIAL_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"
