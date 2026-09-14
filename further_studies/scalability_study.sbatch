#!/bin/bash
# SLURM batch script for scalability study (ThreadsDeployment).

#SBATCH --job-name=Scalability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=1024
#SBATCH --hin=nomultithread

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

python -u further_studies/run_scalability_study.py \
    --stream-length 2000 \
    --drift-frequency 100 \
    --n-repeats 3 \
    --seed 42 \
    --scenario balanced \
    --track-stats \
    --use-jumper \
    --pin-cpus \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG}"
