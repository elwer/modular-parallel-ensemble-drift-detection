#!/bin/bash
#SBATCH --job-name=drift-cv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8192

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/eliasw/modular-parallel-ensemble-drift-detection}"
source setup.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python split_pipeline/run_cv_job.py \
  --mode "${MODE}" \
  --detector-type "${DETECTOR_TYPE}" \
  --fold-config "${FOLD_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --optuna-storage "${OPTUNA_STORAGE}" \
  --stream-length "${STREAM_LENGTH:-8000}" \
  --seed "${SEED:-1337}" \
  --expert-trials "${N_TRIALS_EXPERT:-100}" \
  --deployment-trials "${N_TRIALS_DEPLOYMENT:-100}" \
  --per-trial-timeout "${PER_TRIAL_TIMEOUT:-1200}" \
  --n-jobs "${N_JOBS:-7}" \
  2>&1 | tee "${OUTPUT_DIR}/cv_${MODE}_${DETECTOR_TYPE}.log"
