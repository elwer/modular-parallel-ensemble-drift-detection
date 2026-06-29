#!/usr/bin/env bash
# Tier-3 ablation driver: re-runs the FULL pipeline (N=1 multi-stream Optuna
# + greedy) once per value of a single pinned MOPEDDS-level global. Each run
# is isolated in its own output sub-directory so the resulting achievable-F1
# vs parameter-value curves isolate the effect of that global on the *whole*
# pipeline (Tier 3 honesty: the pool itself is re-built under the pin).
#
# Output layout (one entry shown):
#   ${ABLATION_OUT_DIR}/${KEY}/v=${VALUE}/${TAG}/
#       synthF1ms_${TAG}_N1_S${N}*.csv     (pool from pinned N=1 study)
#       synthF1ms_${TAG}_N1_S${N}_eval.csv
#       greedy_${TAG}_S${N}.csv            (greedy result for this pin value)
#
# Usage:
#   ./submit_global_ablation.sh <KEY> <VALUES_CSV> [GENERATORS_STR]
#
# Examples:
#   ./submit_global_ablation.sh decision_window         1,3,5,8,15           "SineClusters WaveformDrift2"
#   ./submit_global_ablation.sh suppression_window      0,1,3,5,8            SineClusters
#   ./submit_global_ablation.sh detector_decision_criteria any,majority,all  SineClusters
#   ./submit_global_ablation.sh ensemble_decision_criteria any,majority,all  SineClusters
#
# Useful env overrides (all defaults match the production pipeline; for an
# ablation a tighter N=1 budget is usually fine):
#   N1_TIMEOUT=14400     # 4h instead of 24h
#   MAX_N=16
#   N_STREAMS=10  BASE_STREAM_SEED=42  DRIFT_FREQUENCIES=...
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <KEY> <VALUES_CSV> [GENERATORS_STR]" >&2
    echo "  KEY in {detector_decision_criteria, ensemble_decision_criteria," >&2
    echo "          decision_window, suppression_window, recent_samples_size}" >&2
    exit 2
fi

KEY="$1"
VALUES_CSV="$2"
GENERATORS_STR="${3:-SineClusters WaveformDrift2}"

case "${KEY}" in
    detector_decision_criteria|ensemble_decision_criteria) ;;
    decision_window|suppression_window|recent_samples_size) ;;
    *)
        echo "ERROR: unsupported KEY '${KEY}'." >&2
        exit 2
        ;;
esac

IFS=',' read -r -a VALUES <<< "${VALUES_CSV}"
read -r -a GENERATOR_ENTRIES <<< "${GENERATORS_STR}"

# Forwarded to submit_full_greedy_pipeline.sh. Sensible defaults; override
# any of these via env to change the per-pipeline budget for the ablation.
ABLATION_OUT_DIR="${ABLATION_OUT_DIR:-synthetic_multistream_results/ablation_${KEY}}"
N_STREAMS="${N_STREAMS:-10}"
BASE_STREAM_SEED="${BASE_STREAM_SEED:-42}"
DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES:-200,400,500,750,1000,1250,1500,2000,2500,3000}"
STREAM_LENGTH="${STREAM_LENGTH:-10000}"
EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES:-1,4,8}"
TOLERANCES="${TOLERANCES:-}"
OBJECTIVE="${OBJECTIVE:-macro}"
N1_TIMEOUT="${N1_TIMEOUT:-86400}"
MAX_N="${MAX_N:-32}"
TOP_K_OVERALL="${TOP_K_OVERALL:-30}"
TOP_K_PER_TYPE="${TOP_K_PER_TYPE:-5}"
INNER_SEARCH="${INNER_SEARCH:-1}"
STOP_ON_NO_IMPROVE="${STOP_ON_NO_IMPROVE:-0}"
GLOBALS_MODE="${GLOBALS_MODE:-best}"
SEED="${SEED:-1337}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/submit_full_greedy_pipeline.sh"

echo "=================================================================="
echo "Tier-3 ablation"
echo "  Key (pinned)      : ${KEY}"
echo "  Values            : ${VALUES[*]}"
echo "  Generator entries : ${GENERATOR_ENTRIES[*]}"
echo "  Out dir base      : ${ABLATION_OUT_DIR}"
echo "  N=1 timeout       : ${N1_TIMEOUT}s"
echo "  Greedy max N      : ${MAX_N}"
echo "=================================================================="

n_total=0
for v in "${VALUES[@]}"; do
    # Sanitize for filesystem use (categorical values + ints both fine).
    safe_v="$(echo "${v}" | tr -c 'A-Za-z0-9._+-' '_')"
    pin_out_dir="${ABLATION_OUT_DIR}/v=${safe_v}"

    echo
    echo ">>> ${KEY}=${v}  ->  ${pin_out_dir}"

    # Forward everything via env vars; tag derivation inside the pipeline is
    # left as-is, so each entry occupies a clean per-generator sub-directory
    # under v=${safe_v}/.
    N_STREAMS="${N_STREAMS}" \
    BASE_STREAM_SEED="${BASE_STREAM_SEED}" \
    DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES}" \
    STREAM_LENGTH="${STREAM_LENGTH}" \
    EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES}" \
    TOLERANCES="${TOLERANCES}" \
    OBJECTIVE="${OBJECTIVE}" \
    N1_TIMEOUT="${N1_TIMEOUT}" \
    MAX_N="${MAX_N}" \
    TOP_K_OVERALL="${TOP_K_OVERALL}" \
    TOP_K_PER_TYPE="${TOP_K_PER_TYPE}" \
    INNER_SEARCH="${INNER_SEARCH}" \
    STOP_ON_NO_IMPROVE="${STOP_ON_NO_IMPROVE}" \
    GLOBALS_MODE="${GLOBALS_MODE}" \
    SEED="${SEED}" \
    OUT_DIR="${pin_out_dir}" \
    PIN_GLOBALS="${KEY}=${v}" \
        "${PIPELINE}" "${GENERATORS_STR}"

    n_total=$((n_total + 1))
done

echo
echo "Submitted ${n_total} ablation pin-value batch(es) for KEY=${KEY}."
echo "Each batch contains one N=1+greedy pipeline per generator entry."
echo "Results land under ${ABLATION_OUT_DIR}/v=<value>/<tag>/."
