#!/bin/bash
# Submit script for ensemble vs generalist comparison on HPC
#
# Usage:
#   ./submit_ensemble_vs_generalist.sh          # default config
#   N_FOLDS=10 ./submit_ensemble_vs_generalist.sh  # override

set -euo pipefail

# ---- Experiment configuration ----
DRIFT_FREQS="${DRIFT_FREQS:-200,500,1000}"
STREAM_LENGTH="${STREAM_LENGTH:-2000}"
N_BUDGET="${N_BUDGET:-60}"
N_DEPLOY_TRIALS="${N_DEPLOY_TRIALS:-30}"
N_FOLDS="${N_FOLDS:-10}"
N_CPUS="${N_CPUS:-32}"
EXPERT_TIMEOUT="${EXPERT_TIMEOUT:-120}"
GENERALIST_TIMEOUT="${GENERALIST_TIMEOUT:-600}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ensemble_vs_generalist_hpc}"
WALLTIME="${WALLTIME:-72:00:00}"
MEM_PER_CPU="${MEM_PER_CPU:-8192}"

export DRIFT_FREQS STREAM_LENGTH N_BUDGET N_DEPLOY_TRIALS N_FOLDS N_CPUS
export EXPERT_TIMEOUT GENERALIST_TIMEOUT OUTPUT_DIR WALLTIME MEM_PER_CPU

sbatch --export=ALL \
    --job-name=EnsVsGen \
    --cpus-per-task="${N_CPUS}" \
    --time="${WALLTIME}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    ensemble_vs_generalist.sbatch
