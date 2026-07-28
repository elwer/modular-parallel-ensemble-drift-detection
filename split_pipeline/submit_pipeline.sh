#!/bin/bash
# Master submission script for the split expert+generalist pipeline.
#
# Submits 14 independent SLURM jobs (no dependencies):
#   - 7 expert jobs (one per DD type): optimize + evaluate per-DD ensemble
#   - 7 generalist jobs (one per DD type): optimize + evaluate on eval set
#
# After all jobs complete, run compare_results.py to build the cross-DD
# best ensemble and print the comparison table.
#
# Usage:
#   bash split_pipeline/submit_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
OPTUNA_STORAGE="sqlite:///split_pipeline/split_pipeline_optuna.db"
N_STREAMS=10
BASE_STREAM_SEED=42
DRIFT_FREQUENCIES="200,400,500,750,1000,1250,1500,2000,2500,3000"
STREAM_LENGTH=8000
EVAL_STREAM_INDICES="1,4,8"
GENERATORS="SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2,SineClusters,WaveformDrift2"
SEED=1337
N_TRIALS_EXPERT=100
N_TRIALS_DEPLOYMENT=100
PER_TRIAL_TIMEOUT=1200
OUTPUT_DIR="split_pipeline/results"
N_JOBS_EXPERT=7

# Generalist: 7 * 100 detector trials + 100 deployment-equivalent trials = 800
N_PROFILES=7
N_TRIALS_GENERALIST=$((N_PROFILES * N_TRIALS_EXPERT + N_TRIALS_DEPLOYMENT))

DETECTOR_TYPES=("BNDM" "CSDDM" "D3" "IBDD" "OCDD" "SPLL" "UDetect")

# Export shared env vars
export OPTUNA_STORAGE N_STREAMS BASE_STREAM_SEED DRIFT_FREQUENCIES STREAM_LENGTH
export EVAL_STREAM_INDICES GENERATORS SEED N_TRIALS_EXPERT N_TRIALS_DEPLOYMENT PER_TRIAL_TIMEOUT
export OUTPUT_DIR

echo "=== Split Pipeline Submission ==="
echo "  Storage:       ${OPTUNA_STORAGE}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Expert trials: ${N_TRIALS_EXPERT} per profile x detector"
echo "  Deployment trials: ${N_TRIALS_DEPLOYMENT} per DD type (Phase 2)"
echo "  Generalist trials: ${N_TRIALS_GENERALIST} per DD type"
echo ""

# Initialize the Optuna DB schema before submitting any jobs.
# This prevents "table studies already exists" errors when 14 jobs
# start simultaneously and all try to create the SQLite schema.
python -c "
import optuna
optuna.storages.RDBStorage('${OPTUNA_STORAGE}')
print('Optuna DB schema initialized.')
"

# Submit expert optimization+eval jobs (one per DD type)
export N_JOBS="${N_JOBS_EXPERT}"
EXPERT_JOB_IDS=()
for DD in "${DETECTOR_TYPES[@]}"; do
    export DETECTOR_TYPE="${DD}"
    JOB_ID=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/expert_optimize.sbatch")
    EXPERT_JOB_IDS+=("${JOB_ID}")
    echo "Expert ${DD} job: ${JOB_ID}"
done
unset DETECTOR_TYPE N_JOBS

# Submit generalist optimization+eval jobs (one per DD type)
GEN_JOB_IDS=()
for DD in "${DETECTOR_TYPES[@]}"; do
    export DETECTOR_TYPE="${DD}"
    export N_TRIALS="${N_TRIALS_GENERALIST}"
    JOB_ID=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/generalist_optimize.sbatch")
    GEN_JOB_IDS+=("${JOB_ID}")
    echo "Generalist ${DD} job: ${JOB_ID}"
done
unset DETECTOR_TYPE N_TRIALS

echo ""
echo "=== Pipeline submitted (14 independent jobs) ==="
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Job summary:"
for i in "${!DETECTOR_TYPES[@]}"; do
    echo "  Expert ${DETECTOR_TYPES[$i]}:     ${EXPERT_JOB_IDS[$i]}"
done
for i in "${!DETECTOR_TYPES[@]}"; do
    echo "  Generalist ${DETECTOR_TYPES[$i]}: ${GEN_JOB_IDS[$i]}"
done
echo ""
echo "After all jobs complete, run compare_results.py to build"
echo "the cross-DD best ensemble and print the comparison table."
