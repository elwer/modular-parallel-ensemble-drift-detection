#!/bin/bash
# Submit script for expert ensemble evaluation from Optuna DB
# This can run in parallel with the expert optimization job.

set -euo pipefail

# Configuration (must match expert optimization)
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
OUTPUT_DIR="expert_ensemble_eval"
N_JOBS=7
POLL_INTERVAL=60

# Export variables
export OPTUNA_STORAGE N_STREAMS BASE_STREAM_SEED DRIFT_FREQUENCIES STREAM_LENGTH
export EVAL_STREAM_INDICES GENERATORS SEED N_TRIALS_EXPERT PER_TRIAL_TIMEOUT
export OUTPUT_DIR N_JOBS POLL_INTERVAL

sbatch --export=ALL evaluate_from_optuna.sbatch
