#!/bin/bash
# SLURM batch script for scalability study (ThreadsDeployment).

#SBATCH --job-name=Scalability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/eliasw/modular-parallel-ensemble-drift-detection}"
source setup.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:-results_scalability_hpc}"
mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/scalability.log"

python -u run_scalability_study.py \
    --stream-length 200 \
    --drift-frequency 100 \
    --n-repeats 5 \
    --seed 42 \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG}"
