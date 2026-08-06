#!/bin/bash
# Submit script for default configs experiment on HPC
#
# Usage:
#   ./submit_defaults_experiment.sh          # default config
#   N_FOLDS=10 ./submit_defaults_experiment.sh  # override

set -euo pipefail

# ---- Experiment configuration ----
DRIFT_FREQS="${DRIFT_FREQS:-200,400,500,750,1000,1250,1500,2000}"
STREAM_LENGTH="${STREAM_LENGTH:-5000}"
N_DEPLOY_TRIALS="${N_DEPLOY_TRIALS:-50}"
N_FOLDS="${N_FOLDS:-10}"
N_CPUS="${N_CPUS:-16}"
GENERALIST_TIMEOUT="${GENERALIST_TIMEOUT:-600}"
OUTPUT_DIR="${OUTPUT_DIR:-results_defaults_experiment_hpc}"
WALLTIME="${WALLTIME:-72:00:00}"
MEM_PER_CPU="${MEM_PER_CPU:-8192}"

export DRIFT_FREQS STREAM_LENGTH N_DEPLOY_TRIALS N_FOLDS N_CPUS
export GENERALIST_TIMEOUT OUTPUT_DIR WALLTIME MEM_PER_CPU

sbatch --export=ALL \
    --job-name=DefExp \
    --cpus-per-task="${N_CPUS}" \
    --time="${WALLTIME}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    defaults_experiment.sbatch
