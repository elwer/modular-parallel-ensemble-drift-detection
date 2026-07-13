#!/bin/bash
# Master submission script for the split expert+generalist pipeline.
#
# Submits separate SLURM jobs:
#   1. Expert optimization (1 job, parallelized across profiles internally)
#   2. Generalist optimization (7 jobs, one per DD type)
#   3. Evaluation (1 job, depends on all of the above)
#
# Usage:
#   bash submit_pipeline.sh
#
# To resume (skip already-completed Optuna studies, just re-evaluate):
#   RESUME=1 bash submit_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration ──────────────────────────────────────────────────────
OPTUNA_STORAGE="sqlite:///expert_ensemble_optuna.db"
N_STREAMS=10
BASE_STREAM_SEED=42
DRIFT_FREQUENCIES="200,400,500,750,1000,1250,1500,2000,2500,3000"
STREAM_LENGTH=8000
EVAL_STREAM_INDICES="1,4,8"
GENERATORS="SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2"
SEED=1337
N_TRIALS_EXPERT=50
PER_TRIAL_TIMEOUT=1200
OUTPUT_DIR="expert_ensemble_results"
N_JOBS_EXPERT=8

# Generalist: K profiles * N_TRIALS_EXPERT = 7 * 50 = 350
N_PROFILES=7
N_TRIALS_GENERALIST=$((N_PROFILES * N_TRIALS_EXPERT))

DETECTOR_TYPES=("BNDM" "CSDDM" "D3" "IBDD" "OCDD" "SPLL" "UDetect")

# ── Export shared env vars ─────────────────────────────────────────────
export OPTUNA_STORAGE N_STREAMS BASE_STREAM_SEED DRIFT_FREQUENCIES STREAM_LENGTH
export EVAL_STREAM_INDICES GENERATORS SEED N_TRIALS_EXPERT PER_TRIAL_TIMEOUT
export OUTPUT_DIR

echo "=== Split Pipeline Submission ==="
echo "  Storage:       ${OPTUNA_STORAGE}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Expert trials: ${N_TRIALS_EXPERT} per profile×detector"
echo "  Generalist trials: ${N_TRIALS_GENERALIST} per DD type"
echo ""

# ── Submit expert optimization job ─────────────────────────────────────
export N_JOBS="${N_JOBS_EXPERT}"
EXPERT_JOB=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/expert_optimize.sbatch")
echo "Expert optimization job: ${EXPERT_JOB}"
unset N_JOBS

# ── Submit generalist optimization jobs (one per DD type) ──────────────
GEN_JOB_IDS=()
for DD in "${DETECTOR_TYPES[@]}"; do
    export DETECTOR_TYPE="${DD}"
    export N_TRIALS="${N_TRIALS_GENERALIST}"
    JOB_ID=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/generalist_optimize.sbatch")
    GEN_JOB_IDS+=("${JOB_ID}")
    echo "Generalist ${DD} job: ${JOB_ID}"
done
unset DETECTOR_TYPE N_TRIALS

# ── Submit evaluation job (depends on all optimization jobs) ───────────
ALL_DEPS="afterok:${EXPERT_JOB}"
for JID in "${GEN_JOB_IDS[@]}"; do
    ALL_DEPS="${ALL_DEPS}:afterok:${JID}"
done

EVAL_JOB=$(sbatch --parsable --dependency="${ALL_DEPS}" --export=ALL "${SCRIPT_DIR}/evaluate_ensembles.sbatch")
echo "Evaluation job: ${EVAL_JOB} (depends on: ${ALL_DEPS})"

echo ""
echo "=== Pipeline submitted ==="
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Job summary:"
echo "  Expert optimization:  ${EXPERT_JOB}"
for i in "${!DETECTOR_TYPES[@]}"; do
    echo "  Generalist ${DETECTOR_TYPES[$i]}: ${GEN_JOB_IDS[$i]}"
done
echo "  Evaluation:           ${EVAL_JOB}"
