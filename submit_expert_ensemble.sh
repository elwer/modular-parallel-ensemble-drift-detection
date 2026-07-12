#!/bin/bash
# Submit script for expert ensemble optimization

set -euo pipefail

# Configuration
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
N_JOBS=8

# Submit job
export OPTUNA_STORAGE N_STREAMS BASE_STREAM_SEED DRIFT_FREQUENCIES STREAM_LENGTH
export EVAL_STREAM_INDICES GENERATORS SEED N_TRIALS_EXPERT PER_TRIAL_TIMEOUT OUTPUT_DIR N_JOBS

sbatch --export=ALL expert_ensemble_optuna.sbatch
