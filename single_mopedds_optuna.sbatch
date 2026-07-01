#!/bin/bash
# SLURM template for single MOPEDDS member Optuna optimization.
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
#   DETECTOR_TYPE         Detector type to optimize (IBDD, OCDD, D3, SPLL, UDetect, CSDDM, BNDM)
#   N_WORKERS             Number of parallel Optuna workers (default: 64)
#   N_TRIALS              Number of Optuna trials (default: 500)
#   SEED                  Base detector seed
#   OUT_DIR               Output directory

#SBATCH --job-name=SingleMOPEDDS
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/data/horse/ws/s4122485-parallelDD/new/modular-parallel-ensemble-drift-detection}"
source setup.sh

# Optuna uses multiprocessing for parallel trials
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TAG="${STUDY_TAG:-SingleMOPEDDS}"
OUT_DIR="${OUT_DIR:-synthetic_multistream_results_2h}"
N_STREAMS="${N_STREAMS:-10}"
BASE_STREAM_SEED="${BASE_STREAM_SEED:-42}"
DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES:-200,400,500,750,1000,1250,1500,2000,2500,3000}"
STREAM_LENGTH="${STREAM_LENGTH:-8000}"
EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES:-1,4,8}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///optuna.db}"
DETECTOR_TYPE="${DETECTOR_TYPE:-IBDD}"
GENERATORS="${GENERATORS:-SineClusters}"
mkdir -p "${OUT_DIR}/${TAG}"
OUTPUT_CSV="${OUT_DIR}/${TAG}/single_mopedds_${DETECTOR_TYPE}_${TAG}_S${N_STREAMS}.csv"
LOG="${OUT_DIR}/${TAG}/single_mopedds_${DETECTOR_TYPE}_${TAG}_S${N_STREAMS}.log"

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
if [[ -n "${N_TRIALS:-}" ]]; then
    EXTRA_ARGS+=(--n-trials "${N_TRIALS}")
fi

python optimization/single_mopedds_optuna.py \
    --optuna-storage "${OPTUNA_STORAGE}" \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --eval-stream-indices "${EVAL_STREAM_INDICES}" \
    --detector-type "${DETECTOR_TYPE}" \
    --n-workers "${N_WORKERS:-64}" \
    --n-trials "${N_TRIALS:-500}" \
    --seed "${SEED:-1337}" \
    --output-csv "${OUTPUT_CSV}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"
