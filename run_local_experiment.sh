#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate

# --- Config ---
# 4 streams: freq 100, 200, 400, 800 with stream_length=2000
# Drifts per stream: 20, 10, 5, 2.5 -- enough for meaningful detection
# Eval: index 1 (freq 200) -- must NOT match any profile
# Train: indices 0, 2, 3 (freq 100, 400, 800)
# 2 profiles: low_freq (100, index 0) and high_freq (400-800, indices 2,3)
OPTUNA_STORAGE="sqlite:///local_experiment/local_optuna.db"
OUTPUT_DIR="local_experiment/results"
N_STREAMS=4
BASE_STREAM_SEED=42
DRIFT_FREQUENCIES="100,200,400,800"
STREAM_LENGTH=2000
EVAL_STREAM_INDICES="1"
GENERATORS="SineClusters,SineClusters,SineClusters,SineClusters"
SEED=1337
N_TRIALS_EXPERT=30
N_TRIALS_DEPLOYMENT=10
PER_TRIAL_TIMEOUT=60
N_JOBS_EXPERT=2
PROFILES='[{"name":"low_freq","generator_filter":"SineClusters","drift_freq_min":100,"drift_freq_max":100},{"name":"high_freq","generator_filter":"SineClusters","drift_freq_min":400,"drift_freq_max":800}]'
N_PROFILES=2
N_TRIALS_GENERALIST=$((N_PROFILES * N_TRIALS_EXPERT + N_TRIALS_DEPLOYMENT))

# Use only 3 fast detector types to keep runtime manageable
DETECTOR_TYPES=("OCDD" "IBDD" "SPLL")

# --- Clean state ---
rm -rf local_experiment
mkdir -p "${OUTPUT_DIR}"

# --- Init Optuna DB ---
python -c "import optuna; optuna.storages.RDBStorage('${OPTUNA_STORAGE}')"

echo "=== Local Experiment ==="
echo "  Streams: ${N_STREAMS}, Length: ${STREAM_LENGTH}, Trials: ${N_TRIALS_EXPERT}"
echo "  Profiles: ${N_PROFILES}, Eval: ${EVAL_STREAM_INDICES}"
echo "  Generalist trials: ${N_TRIALS_GENERALIST}"
echo ""

# --- Phase 1: Expert optimization (sequential, n_jobs=3 parallel profiles) ---
for DD in "${DETECTOR_TYPES[@]}"; do
    echo ">>> Expert ${DD}..."
    python split_pipeline/expert_optimize.py \
        --optuna-storage "${OPTUNA_STORAGE}" \
        --detector-type "${DD}" \
        --n-streams "${N_STREAMS}" \
        --base-stream-seed "${BASE_STREAM_SEED}" \
        --drift-frequencies "${DRIFT_FREQUENCIES}" \
        --stream-length "${STREAM_LENGTH}" \
        --eval-stream-indices "${EVAL_STREAM_INDICES}" \
        --generators "${GENERATORS}" \
        --seed "${SEED}" \
        --n-trials-expert "${N_TRIALS_EXPERT}" \
        --n-trials-deployment "${N_TRIALS_DEPLOYMENT}" \
        --per-trial-timeout "${PER_TRIAL_TIMEOUT}" \
        --output-dir "${OUTPUT_DIR}" \
        --n-jobs "${N_JOBS_EXPERT}" \
        --load-if-exists \
        --profiles "${PROFILES}" \
        > "${OUTPUT_DIR}/expert_${DD}.log" 2>&1
    echo "<<< Expert ${DD} done"
done

# --- Phase 2: Generalist optimization (2 in parallel) ---
PIDS=()
for DD in "${DETECTOR_TYPES[@]}"; do
    echo ">>> Generalist ${DD}..."
    python split_pipeline/generalist_optimize.py \
        --optuna-storage "${OPTUNA_STORAGE}" \
        --n-streams "${N_STREAMS}" \
        --base-stream-seed "${BASE_STREAM_SEED}" \
        --drift-frequencies "${DRIFT_FREQUENCIES}" \
        --stream-length "${STREAM_LENGTH}" \
        --eval-stream-indices "${EVAL_STREAM_INDICES}" \
        --generators "${GENERATORS}" \
        --detector-type "${DD}" \
        --seed "${SEED}" \
        --n-trials "${N_TRIALS_GENERALIST}" \
        --per-trial-timeout "${PER_TRIAL_TIMEOUT}" \
        --output-dir "${OUTPUT_DIR}" \
        --load-if-exists \
        --profiles "${PROFILES}" \
        > "${OUTPUT_DIR}/generalist_${DD}.log" 2>&1 &
    PIDS+=($!)
    # Run 2 at a time
    if [[ ${#PIDS[@]} -ge 2 ]]; then
        for pid in "${PIDS[@]}"; do wait "${pid}"; done
        PIDS=()
    fi
done
# Wait for any remaining
for pid in "${PIDS[@]}"; do wait "${pid}"; done

echo "<<< All generalists done"

# --- Phase 3: Compare results ---
echo ""
echo "=== Comparison ==="
python split_pipeline/compare_results.py \
    --optuna-storage "${OPTUNA_STORAGE}" \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --eval-stream-indices "${EVAL_STREAM_INDICES}" \
    --generators "${GENERATORS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --profiles "${PROFILES}" \
    2>&1 | tee "${OUTPUT_DIR}/comparison.log"

echo ""
echo "=== Experiment complete ==="
echo "Results in ${OUTPUT_DIR}/"
