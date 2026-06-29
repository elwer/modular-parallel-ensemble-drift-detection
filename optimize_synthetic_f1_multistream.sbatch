#!/bin/bash
# SLURM template for one (generator, ensemble_size) multi-stream Optuna
# F1 optimization run. Submitted by
# submit_synthetic_f1_multistream_optimization.sh.
#
# Required env vars (set via sbatch --export):
#   GENERATOR        Generator class name: SineClusters | WaveformDrift2
#   SIZE             Ensemble size (1, 2, 4, 8, 16, 32, 64 or 128)
#   N_STREAMS        Number of streams to evaluate per trial (e.g. 10)
#   BASE_STREAM_SEED Base seed for stream generation (fixed across all runs)
#   DRIFT_FREQUENCIES Comma-separated per-stream drift frequencies
#                     (length 1 or N_STREAMS)
#   STREAM_LENGTH    Stream length in samples
#   TIMEOUT          Optuna wall-clock budget in seconds (e.g. 86400 = 24h)
#   EVAL_STREAM_INDICES  Optional comma-separated indices into the full stream
#                    list used as the HELD-OUT eval subset. If unset, Python
#                    defaults to "1,4,8" (drift-frequency-stratified).
#   TOLERANCES       Optional comma-separated per-stream TP-matching tolerance
#                    windows. If unset, each defaults to
#                    max(1, drift_frequency // 50) (~2% of inter-drift gap).
#   OBJECTIVE        Optional aggregation across train streams used as the
#                    Optuna objective: 'macro' (default) or 'micro'.
#   OUT_DIR          Output directory
#   SEED             Base seed for sampler / detectors

#SBATCH --job-name=SynthF1msOpt
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

cd /data/horse/ws/s4122485-parallelDD/modular-parallel-ensemble-drift-detection
source setup.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

# Tag used for subdirectory + log/CSV file names. Falls back to GENERATOR when
# STUDY_TAG is unset (single-generator legacy runs).
TAG="${STUDY_TAG:-${GENERATOR}}"
mkdir -p "${OUT_DIR}/${TAG}"
LOG="${OUT_DIR}/${TAG}/synthF1ms_${TAG}_N${SIZE}_S${N_STREAMS}.log"

# Resolve which selector to pass: GENERATORS (per-stream list) takes
# precedence over GENERATOR (single class, legacy). Either MUST be set.
EXTRA_ARGS=()
if [[ -n "${GENERATORS:-}" ]]; then
    EXTRA_ARGS+=(--generators "${GENERATORS}")
elif [[ -n "${GENERATOR:-}" ]]; then
    EXTRA_ARGS+=(--generator "${GENERATOR}")
else
    echo "ERROR: must set GENERATOR (single) or GENERATORS (per-stream list)." >&2
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
if [[ -n "${OBJECTIVE:-}" ]]; then
    EXTRA_ARGS+=(--objective "${OBJECTIVE}")
fi

python optimization/synthetic_f1_multistream_optimize_optuna.py \
    --size "${SIZE}" \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --timeout "${TIMEOUT}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}/${TAG}" \
    --n-workers "${N_WORKERS}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"
