#!/bin/bash
# SLURM template for one greedy-ensemble-from-pool run.
# Submitted by submit_greedy_ensemble.sh.
#
# Required env vars (set via sbatch --export):
#   GENERATOR             Single class (legacy). EITHER this or GENERATORS.
#   GENERATORS            Comma-separated per-stream list (length 1 or N_STREAMS).
#                         If set it overrides GENERATOR.
#   STUDY_TAG             Output tag (subdir + CSV name). Defaults to GENERATOR
#                         when homogeneous, otherwise required for mixed runs.
#   N_STREAMS             Number of streams (must match the N=1 study)
#   BASE_STREAM_SEED      Same as the N=1 study
#   DRIFT_FREQUENCIES     Same as the N=1 study
#   STREAM_LENGTH         Same as the N=1 study
#   EVAL_STREAM_INDICES   Same as the N=1 study (held-out indices)
#   TOLERANCES            Same as the N=1 study (empty -> default)
#   POOL_GLOB             Glob expanding to N=1 per-trial CSVs
#   TOP_K_OVERALL         e.g. 30
#   TOP_K_PER_TYPE        e.g. 5
#   MAX_N                 e.g. 32
#   INNER_SEARCH          "1" to enable --inner-search-ens-crit
#   STOP_ON_NO_IMPROVE    "1" to enable --stop-on-no-improve
#   GLOBALS_MODE          'best' or 'manual'
#   SEED                  Base detector seed (1337 if matching default)
#   OUT_DIR               e.g. synthetic_multistream_results

#SBATCH --job-name=GreedyEns
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/data/horse/ws/s4122485-parallelDD/new/modular-parallel-ensemble-drift-detection}"
source setup.sh

# MOPEDDS run_ensemble is sequential; OMP/BLAS settings don't affect it
# Keep single-threaded to avoid resource contention
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TAG="${STUDY_TAG:-${GENERATOR}}"
mkdir -p "${OUT_DIR}/${TAG}"
OUTPUT_CSV="${OUT_DIR}/${TAG}/greedy_${TAG}_S${N_STREAMS}.csv"
LOG="${OUT_DIR}/${TAG}/greedy_${TAG}_S${N_STREAMS}.log"

EXTRA_ARGS=()
if [[ -n "${GENERATORS:-}" ]]; then
    EXTRA_ARGS+=(--generators "${GENERATORS}")
elif [[ -n "${GENERATOR:-}" ]]; then
    EXTRA_ARGS+=(--generator "${GENERATOR}")
else
    echo "ERROR: must set GENERATOR or GENERATORS." >&2
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
if [[ "${INNER_SEARCH:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--inner-search-ens-crit)
fi
if [[ "${STOP_ON_NO_IMPROVE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--stop-on-no-improve)
fi
if [[ -n "${PIN_GLOBALS:-}" ]]; then
    EXTRA_ARGS+=(--pin-globals "${PIN_GLOBALS}")
fi
if [[ -n "${N_WORKERS:-}" ]]; then
    EXTRA_ARGS+=(--n-workers "${N_WORKERS}")
fi
if [[ -n "${SELECTION_STRATEGY:-}" ]]; then
    EXTRA_ARGS+=(--selection-strategy "${SELECTION_STRATEGY}")
fi
if [[ -n "${INNER_SEARCH_DECISION_WINDOW:-}" ]]; then
    EXTRA_ARGS+=(--inner-search-decision-window "${INNER_SEARCH_DECISION_WINDOW}")
fi
if [[ -n "${INNER_SEARCH_SUPPRESSION_WINDOW:-}" ]]; then
    EXTRA_ARGS+=(--inner-search-suppression-window "${INNER_SEARCH_SUPPRESSION_WINDOW}")
fi
if [[ "${INNER_SEARCH_DET_CRIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--inner-search-det-crit)
fi

python optimization/greedy_ensemble_from_pool.py \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --pool-glob "${POOL_GLOB}" \
    --top-k-overall "${TOP_K_OVERALL}" \
    --top-k-per-type "${TOP_K_PER_TYPE}" \
    --max-n "${MAX_N}" \
    --globals "${GLOBALS_MODE}" \
    --seed "${SEED}" \
    --output-csv "${OUTPUT_CSV}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"
