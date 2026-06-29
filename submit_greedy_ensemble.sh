#!/usr/bin/env bash
# Submit one greedy-ensemble-from-pool job per generator.
# Reuses the SAME stream set + train/eval split as the multistream Optuna
# study so the resulting greedy_<gen>_S<N>.csv is directly comparable to
# the joint-search synthF1ms_<gen>_N<size>_S<N>.csv + _eval.csv files.
set -euo pipefail

GENERATORS_STR="${1:-SineClusters WaveformDrift2}"
read -r -a GENERATORS <<< "${GENERATORS_STR}"

# Entries follow the same convention as submit_synthetic_f1_multistream_optimization.sh:
#   'SineClusters'                            -> homogeneous study, tag=SineClusters
#   'Mix:SineClusters,WaveformDrift2,...'     -> per-stream list of length 1 or N_STREAMS,
#                                                tag=Mix_<sorted-unique-joined-by-+>
#
# POOL_SOURCE_TAG (optional, default = tag of this entry) selects which N=1
# study to read pool entries from. Set it explicitly to mix pools from a
# different study (e.g. read the homogeneous SineClusters + WaveformDrift2
# pools for a mixed greedy run).

# These MUST match what was used for the N=1 multistream study; otherwise the
# pool entries were tuned on a different problem.
N_STREAMS="${N_STREAMS:-10}"
BASE_STREAM_SEED="${BASE_STREAM_SEED:-42}"
DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES:-200,400,500,750,1000,1250,1500,2000,2500,3000}"
STREAM_LENGTH="${STREAM_LENGTH:-10000}"
EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES:-1,4,8}"
TOLERANCES="${TOLERANCES:-}"

SEED="${SEED:-1337}"
OUT_DIR="${OUT_DIR:-synthetic_multistream_results}"

# Pool selection.
TOP_K_OVERALL="${TOP_K_OVERALL:-30}"
TOP_K_PER_TYPE="${TOP_K_PER_TYPE:-5}"
MAX_N="${MAX_N:-32}"
INNER_SEARCH="${INNER_SEARCH:-1}"
STOP_ON_NO_IMPROVE="${STOP_ON_NO_IMPROVE:-0}"
GLOBALS_MODE="${GLOBALS_MODE:-best}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/greedy_ensemble.sbatch"

n_jobs=0
for entry in "${GENERATORS[@]}"; do
    # Resolve (GEN_ENV, GEN_LIST_ENV, TAG, default pool source tag).
    if [[ "${entry}" == Mix:* ]]; then
        gens_list="${entry#Mix:}"
        tag="Mix_$(echo "${gens_list}" | tr ',' '\n' | sort -u | paste -sd '+' -)"
        GEN_ENV=""
        GEN_LIST_ENV="${gens_list}"
        STUDY_TAG_ENV="${tag}"
    else
        GEN_ENV="${entry}"
        GEN_LIST_ENV=""
        STUDY_TAG_ENV=""
        tag="${entry}"
    fi

    # Pool source defaults to this entry's tag, but can be overridden so a
    # mixed run can draw its pool from the per-generator N=1 studies.
    pool_tag="${POOL_SOURCE_TAG:-${tag}}"

    mkdir -p "${OUT_DIR}/${tag}"
    POOL_GLOB="${OUT_DIR}/${pool_tag}/synthF1ms_${pool_tag}_N1_S${N_STREAMS}*.csv"

    echo "Submitting greedy ${tag}  (S=${N_STREAMS}, max_n=${MAX_N})"
    echo "  pool glob = ${POOL_GLOB}"

    GENERATOR="${GEN_ENV}" \
    GENERATORS="${GEN_LIST_ENV}" \
    STUDY_TAG="${STUDY_TAG_ENV}" \
    N_STREAMS="${N_STREAMS}" \
    BASE_STREAM_SEED="${BASE_STREAM_SEED}" \
    DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES}" \
    STREAM_LENGTH="${STREAM_LENGTH}" \
    EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES}" \
    TOLERANCES="${TOLERANCES}" \
    POOL_GLOB="${POOL_GLOB}" \
    TOP_K_OVERALL="${TOP_K_OVERALL}" \
    TOP_K_PER_TYPE="${TOP_K_PER_TYPE}" \
    MAX_N="${MAX_N}" \
    INNER_SEARCH="${INNER_SEARCH}" \
    STOP_ON_NO_IMPROVE="${STOP_ON_NO_IMPROVE}" \
    GLOBALS_MODE="${GLOBALS_MODE}" \
    SEED="${SEED}" \
    OUT_DIR="${OUT_DIR}" \
    sbatch \
        --job-name="Greedy_${tag}" \
        --export=ALL \
        "${SBATCH_TEMPLATE}"
    n_jobs=$((n_jobs + 1))
    sleep 1
done

echo "Submitted ${n_jobs} jobs."
