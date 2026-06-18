#!/bin/bash
# SLURM template for one (dataset, ensemble_size) Optuna F1 optimization run.
# Submitted by submit_synthetic_f1_optimization.sh.
#
# Required env vars (set via sbatch --export):
#   DATASET     Dataset class expression, e.g. "SineClustersPre()"
#   SIZE        Ensemble size (1, 2, 4, 8, 16, 32, 64 or 128)
#   TIMEOUT     Optuna wall-clock budget in seconds (e.g. 86400 = 24h)
#   TOLERANCE   TP-matching tolerance (samples)
#   OUT_DIR     Output directory for CSV / sqlite study
#   SEED        Base seed

#SBATCH --job-name=SynthF1Opt
#SBATCH --cpu-freq=2000000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

cd /data/horse/ws/s4122485-parallelDD/modular-parallel-ensemble-drift-detection
source setup.sh

dataset_label="${DATASET%%(*}"
mkdir -p "${OUT_DIR}/${dataset_label}"
LOG="${OUT_DIR}/${dataset_label}/synthF1_${dataset_label}_N${SIZE}.log"

python optimization/synthetic_f1_optimize_optuna.py \
    --dataset "${DATASET}" \
    --size "${SIZE}" \
    --timeout "${TIMEOUT}" \
    --tolerance "${TOLERANCE}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}/${dataset_label}" \
    2>&1 | tee "${LOG}"
