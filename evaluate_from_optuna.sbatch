#!/bin/bash
#SBATCH --job-name=expert_eval
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=expert_ensemble_eval/log_%j.out
#SBATCH --error=expert_ensemble_eval/log_%j.err

set -euo pipefail

# Load environment variables (same as expert optimization)
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///expert_ensemble_optuna.db}"
N_STREAMS="${N_STREAMS:-10}"
BASE_STREAM_SEED="${BASE_STREAM_SEED:-42}"
DRIFT_FREQUENCIES="${DRIFT_FREQUENCIES:-200,400,500,750,1000,1250,1500,2000,2500,3000}"
STREAM_LENGTH="${STREAM_LENGTH:-8000}"
EVAL_STREAM_INDICES="${EVAL_STREAM_INDICES:-1,4,8}"
SEED="${SEED:-1337}"
N_TRIALS_EXPERT="${N_TRIALS_EXPERT:-50}"
PER_TRIAL_TIMEOUT="${PER_TRIAL_TIMEOUT:-1200}"
OUTPUT_DIR="${OUTPUT_DIR:-expert_ensemble_eval}"
N_JOBS="${N_JOBS:-7}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# Generators (must match expert optimization)
GENERATORS="${GENERATORS:-SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2}"

# Enable live output
export PYTHONUNBUFFERED=1

# Limit BLAS threads to avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${OUTPUT_DIR}"

LOG="${OUTPUT_DIR}/eval.log"

echo "=== Expert Ensemble Evaluation from Optuna DB ==="
echo "Optuna storage: ${OPTUNA_STORAGE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "N jobs: ${N_JOBS}"
echo "Started at: $(date)"
echo ""

python optimization/evaluate_from_optuna.py \
    --optuna-storage "${OPTUNA_STORAGE}" \
    --n-streams "${N_STREAMS}" \
    --base-stream-seed "${BASE_STREAM_SEED}" \
    --drift-frequencies "${DRIFT_FREQUENCIES}" \
    --stream-length "${STREAM_LENGTH}" \
    --eval-stream-indices "${EVAL_STREAM_INDICES}" \
    --generators "${GENERATORS}" \
    --seed "${SEED}" \
    --n-trials-expert "${N_TRIALS_EXPERT}" \
    --per-trial-timeout "${PER_TRIAL_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-jobs "${N_JOBS}" \
    --wait \
    --poll-interval "${POLL_INTERVAL}" \
    2>&1 | tee "${LOG}"

echo ""
echo "Finished at: $(date)"
