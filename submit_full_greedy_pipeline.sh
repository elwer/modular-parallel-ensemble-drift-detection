#!/usr/bin/env bash
# End-to-end driver for the greedy ensemble experiment.
#
# For each generator entry:
#   1. If the N=1 multi-stream Optuna pool CSV(s) for this tag already exist
#      (non-empty), reuse them.
#   2. Otherwise submit the N=1 multi-stream optimization job (size=1 only)
#      and capture its SLURM jobid.
#   3. Submit the greedy-ensemble job with --dependency=afterok:<jobid> when
#      step 2 ran, so it starts only once the pool is ready.
#
# Generator entries follow the same convention as submit_synthetic_f1_multistream_optimization.sh:
#   'SineClusters'                              -> homogeneous study, tag=SineClusters
#   'Mix:SineClusters,WaveformDrift2,...'       -> per-stream list of length 1 or N_STREAMS,
#                                                  tag=Mix_<sorted-unique-joined-by-+>
#
# Override FORCE_POOL=1 to re-run N=1 even when CSVs already exist.
set -euo pipefail

GENERATORS_STR="${1:-SineClusters WaveformDrift2}"
read -r -a GENERATORS <<< "${GENERATORS_STR}"

# -- Stream set + split (must match between N=1 study and greedy run). -------
N_STREAMS="${N_STREAMS:-10}"
BASE_STREAM_SEED="${BASE_STREAM_SEED:-42}"
DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES:-200,400,500,750,1000,1250,1500,2000,2500,3000}"
STREAM_LENGTH="${STREAM_LENGTH:-10000}"
EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES:-1,4,8}"
TOLERANCES="${TOLERANCES:-}"
OBJECTIVE="${OBJECTIVE:-macro}"
SEED="${SEED:-1337}"
OUT_DIR="${OUT_DIR:-synthetic_multistream_results}"

# -- N=1 study budget. -------------------------------------------------------
N1_TIMEOUT="${N1_TIMEOUT:-86400}"   # Optuna wall-clock for the N=1 study (sec).

# -- Greedy stage knobs. -----------------------------------------------------
TOP_K_OVERALL="${TOP_K_OVERALL:-30}"
TOP_K_PER_TYPE="${TOP_K_PER_TYPE:-5}"
MAX_N="${MAX_N:-32}"
INNER_SEARCH="${INNER_SEARCH:-1}"
STOP_ON_NO_IMPROVE="${STOP_ON_NO_IMPROVE:-0}"
GLOBALS_MODE="${GLOBALS_MODE:-best}"

# Greedy can draw its pool from a tag OTHER than the one its evaluation set
# uses. Default: same tag as the greedy stage.
POOL_SOURCE_TAG_OVERRIDE="${POOL_SOURCE_TAG:-}"

FORCE_POOL="${FORCE_POOL:-0}"

# Pin one or more MOPEDDS globals to fixed values for both the N=1 study and
# the greedy stage. Pass as 'key1=val1,key2=val2'. Used by the ablation driver.
PIN_GLOBALS="${PIN_GLOBALS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N1_SBATCH="${SCRIPT_DIR}/optimize_synthetic_f1_multistream.sbatch"
GREEDY_SBATCH="${SCRIPT_DIR}/greedy_ensemble.sbatch"

_resolve_entry () {
    # Echo: '<tag>|<gen_env>|<gen_list_env>|<study_tag_env>'
    local entry="$1"
    if [[ "${entry}" == Mix:* ]]; then
        local gens_list="${entry#Mix:}"
        local tag
        tag="Mix_$(echo "${gens_list}" | tr ',' '\n' | sort -u | paste -sd '+' -)"
        printf '%s|%s|%s|%s\n' "${tag}" "" "${gens_list}" "${tag}"
    else
        printf '%s|%s|%s|%s\n' "${entry}" "${entry}" "" ""
    fi
}

_pool_csvs_exist () {
    # Return 0 (success) iff at least one matching CSV is present AND
    # non-empty under ${OUT_DIR}/${1}/synthF1ms_${1}_N1_S${N_STREAMS}*.csv .
    local tag="$1"
    local pattern="${OUT_DIR}/${tag}/synthF1ms_${tag}_N1_S${N_STREAMS}*.csv"
    shopt -s nullglob
    local files=( ${pattern} )
    shopt -u nullglob
    for f in "${files[@]}"; do
        # Skip the *_eval.csv aggregate -- per-trial CSVs only.
        if [[ "${f}" == *_eval.csv ]]; then
            continue
        fi
        if [[ -s "${f}" ]]; then
            return 0
        fi
    done
    return 1
}

n_total=0
n_n1=0
n_greedy=0
for entry in "${GENERATORS[@]}"; do
    IFS='|' read -r tag GEN_ENV GEN_LIST_ENV STUDY_TAG_ENV <<< "$(_resolve_entry "${entry}")"

    mkdir -p "${OUT_DIR}/${tag}"
    pool_tag="${POOL_SOURCE_TAG_OVERRIDE:-${tag}}"
    POOL_GLOB="${OUT_DIR}/${pool_tag}/synthF1ms_${pool_tag}_N1_S${N_STREAMS}*.csv"

    # ------------------------------------------------------------------ N=1
    DEP_FLAG=""
    if [[ "${FORCE_POOL}" != "1" ]] && _pool_csvs_exist "${pool_tag}"; then
        echo "[${tag}] pool CSVs already exist under ${OUT_DIR}/${pool_tag}/ -> skipping N=1 study."
    else
        if [[ -n "${POOL_SOURCE_TAG_OVERRIDE}" && "${POOL_SOURCE_TAG_OVERRIDE}" != "${tag}" ]]; then
            echo "[${tag}] WARNING: pool source tag '${POOL_SOURCE_TAG_OVERRIDE}' differs from this entry's tag and no CSVs exist for it."
            echo "          Submitting an N=1 study under the OVERRIDE tag is not automatic; run that study manually first or"
            echo "          unset POOL_SOURCE_TAG and re-run."
            exit 2
        fi
        echo "[${tag}] submitting N=1 multistream study (size=1, timeout=${N1_TIMEOUT}s)"
        JID=$(GENERATOR="${GEN_ENV}" \
              GENERATORS="${GEN_LIST_ENV}" \
              STUDY_TAG="${STUDY_TAG_ENV}" \
              SIZE=1 \
              N_STREAMS="${N_STREAMS}" \
              BASE_STREAM_SEED="${BASE_STREAM_SEED}" \
              DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES}" \
              STREAM_LENGTH="${STREAM_LENGTH}" \
              EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES}" \
              TOLERANCES="${TOLERANCES}" \
              OBJECTIVE="${OBJECTIVE}" \
              TIMEOUT="${N1_TIMEOUT}" \
              SEED="${SEED}" \
              OUT_DIR="${OUT_DIR}" \
              PIN_GLOBALS="${PIN_GLOBALS}" \
              sbatch --parsable \
                  --job-name="SynthF1ms_${tag}_N1" \
                  --export=ALL \
                  "${N1_SBATCH}")
        echo "  -> N=1 jobid ${JID}"
        DEP_FLAG="--dependency=afterok:${JID}"
        n_n1=$((n_n1 + 1))
        sleep 1
    fi

    # ------------------------------------------------------------------ greedy
    echo "[${tag}] submitting greedy stage (max_n=${MAX_N}, pool_glob=${POOL_GLOB})"
    GJID=$(GENERATOR="${GEN_ENV}" \
           GENERATORS="${GEN_LIST_ENV}" \
           STUDY_TAG="${STUDY_TAG_ENV}" \
           PIN_GLOBALS="${PIN_GLOBALS}" \
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
           sbatch --parsable \
               ${DEP_FLAG} \
               --job-name="Greedy_${tag}" \
               --export=ALL \
               "${GREEDY_SBATCH}")
    echo "  -> Greedy jobid ${GJID}${DEP_FLAG:+ (waits on ${DEP_FLAG#--dependency=})}"
    n_greedy=$((n_greedy + 1))
    n_total=$((n_total + 1))
    sleep 1
done

echo
echo "Submitted ${n_total} pipeline(s): ${n_n1} N=1 study job(s) + ${n_greedy} greedy job(s)."
