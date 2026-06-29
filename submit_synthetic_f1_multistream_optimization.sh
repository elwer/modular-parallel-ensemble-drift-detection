#!/usr/bin/env bash
# Submit one 24h multi-stream Optuna job per (generator, ensemble size).
# Each trial evaluates the SAME N streams (fixed seeds) split into a TRAIN
# subset (used by Optuna) and a fixed HELD-OUT EVAL subset. The objective is
# the MACRO MEAN of per-stream F1 over the train subset (configurable to
# MICRO via OBJECTIVE=micro). The best trial is then re-evaluated on the eval
# subset and the result is appended to
# <OUT_DIR>/<GENERATOR>/synthF1ms_<GEN>_N<size>_S<N>_eval.csv.
set -euo pipefail

GENERATORS_STR="${1:-SineClusters WaveformDrift2}"
SIZES_STR="${2:-1 2 4 8 16 32 64 128}"
read -r -a GENERATORS <<< "${GENERATORS_STR}"
read -r -a SIZES <<< "${SIZES_STR}"

# Each entry in GENERATORS is either a single generator class (e.g.
# 'SineClusters') -- in which case all N streams use that one generator -- or
# a name of the form 'Mix:<g1>,<g2>,...' (a per-stream list of length 1 or
# N_STREAMS). For 'Mix:'-style entries, the substring after the colon is
# passed through to --generators verbatim. The submit script picks the study
# tag (used in the output sub-directory and CSV names) automatically.

N_STREAMS="${N_STREAMS:-10}"
BASE_STREAM_SEED="${BASE_STREAM_SEED:-42}"
# Per-stream drift frequencies (must have length 1 or N_STREAMS).
DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES:-200,400,500,750,1000,1250,1500,2000,2500,3000}"
STREAM_LENGTH="${STREAM_LENGTH:-10000}"
# Held-out evaluation indices into the full N-stream list. Must be the SAME
# across every (size, generator) run for the results to be comparable.
# Default '1,4,8' stratifies the split by drift frequency: with the default
# DRIFT_FREQUENCIES [200,400,500,750,1000,1250,1500,2000,2500,3000] this
# yields eval df {400,1000,2500} (mean 1300) vs train df
# {200,500,750,1250,1500,2000,3000} (mean 1314). For F1 the absolute
# subset means matter less (F1 is bounded in [0,1]) but a stratified split
# still ensures the per-stream tolerance distribution is comparable across
# train and eval.
EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES:-1,4,8}"
# Per-stream TP-matching tolerances (samples). Empty -> Python default
# max(1, drift_frequency // 50) per stream (~2% of inter-drift gap). Set to
# a single value to broadcast it to all N streams, or a comma-separated list
# of length N_STREAMS for per-stream control.
TOLERANCES="${TOLERANCES:-}"
# Aggregation across train streams used as the Optuna objective: 'macro'
# (mean of per-stream F1, default) or 'micro' (F1 of summed TP/FP/FN).
OBJECTIVE="${OBJECTIVE:-macro}"

TIMEOUT="${TIMEOUT:-86400}"
SEED="${SEED:-1337}"
OUT_DIR="${OUT_DIR:-synthetic_multistream_results}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/optimize_synthetic_f1_multistream.sbatch"

n_jobs=0
for entry in "${GENERATORS[@]}"; do
    # Resolve (GEN_ENV, GEN_LIST_ENV, TAG) from the entry.
    if [[ "${entry}" == Mix:* ]]; then
        # Mixed-generator study: 'Mix:<csv list>' -> GENERATORS=<csv list>.
        gens_list="${entry#Mix:}"
        # Build tag = 'Mix_' + sorted unique generator names joined by '+'.
        tag="Mix_$(echo "${gens_list}" | tr ',' '\n' | sort -u | paste -sd '+' -)"
        GEN_ENV=""
        GEN_LIST_ENV="${gens_list}"
        STUDY_TAG_ENV="${tag}"
    else
        # Single-generator study (legacy form).
        GEN_ENV="${entry}"
        GEN_LIST_ENV=""
        STUDY_TAG_ENV=""
        tag="${entry}"
    fi
    mkdir -p "${OUT_DIR}/${tag}"
    for size in "${SIZES[@]}"; do
        echo "Submitting ${tag} N=${size} (S=${N_STREAMS} streams)"
        GENERATOR="${GEN_ENV}" \
        GENERATORS="${GEN_LIST_ENV}" \
        STUDY_TAG="${STUDY_TAG_ENV}" \
        SIZE="${size}" \
        N_STREAMS="${N_STREAMS}" \
        BASE_STREAM_SEED="${BASE_STREAM_SEED}" \
        DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES}" \
        STREAM_LENGTH="${STREAM_LENGTH}" \
        EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES}" \
        TOLERANCES="${TOLERANCES}" \
        OBJECTIVE="${OBJECTIVE}" \
        TIMEOUT="${TIMEOUT}" \
        SEED="${SEED}" \
        OUT_DIR="${OUT_DIR}" \
        sbatch \
            --job-name="SynthF1ms_${tag}_N${size}" \
            --export=ALL \
            "${SBATCH_TEMPLATE}"
        n_jobs=$((n_jobs + 1))
        sleep 1
    done
done

echo "Submitted ${n_jobs} jobs."
