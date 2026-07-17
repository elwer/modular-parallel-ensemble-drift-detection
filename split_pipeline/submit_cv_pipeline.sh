#!/bin/bash
# Submit a stratified repeated cross-validation experiment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${CONFIG_DIR:-${ROOT_DIR}/split_pipeline/cv_configs}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/split_pipeline/cv_results}"
N_REPEATS="${N_REPEATS:-3}"
BASE_SEED="${BASE_SEED:-42}"
BASE_STREAM_SEED="${BASE_SEED}"
STREAM_LENGTH="${STREAM_LENGTH:-8000}"
N_TRIALS_EXPERT="${N_TRIALS_EXPERT:-100}"
N_TRIALS_DEPLOYMENT="${N_TRIALS_DEPLOYMENT:-100}"
PER_TRIAL_TIMEOUT="${PER_TRIAL_TIMEOUT:-1200}"
N_JOBS="${N_JOBS:-7}"
SEED="${SEED:-1337}"
OPTUNA_STORAGE_ROOT="${OPTUNA_STORAGE_ROOT:-${ROOT_DIR}/split_pipeline/cv_optuna}"
mkdir -p "${CONFIG_DIR}" "${RESULT_DIR}" "${OPTUNA_STORAGE_ROOT}"

python "${ROOT_DIR}/split_pipeline/generate_cv_configs.py" \
  --output-dir "${CONFIG_DIR}" --repeats "${N_REPEATS}" --base-seed "${BASE_SEED}"

DETECTORS=(BNDM CSDDM D3 IBDD OCDD SPLL UDetect)
for fold in $(seq 0 $((N_REPEATS * 5 - 1))); do
  CONFIG="${CONFIG_DIR}/fold_${fold}.json"
  for detector in "${DETECTORS[@]}"; do
    for mode in expert generalist; do
      output="${RESULT_DIR}/fold_${fold}/${detector}/${mode}"
      mkdir -p "${output}"
      storage="sqlite:///${OPTUNA_STORAGE_ROOT}/fold_${fold}_${detector}_${mode}.db"
      export MODE="${mode}" DETECTOR_TYPE="${detector}" FOLD_CONFIG="${CONFIG}"
      export OUTPUT_DIR="${output}" OPTUNA_STORAGE="${storage}"
      export STREAM_LENGTH BASE_STREAM_SEED N_TRIALS_EXPERT N_TRIALS_DEPLOYMENT PER_TRIAL_TIMEOUT N_JOBS SEED
      sbatch --parsable --export=ALL "${ROOT_DIR}/split_pipeline/cv_job.sbatch"
    done
  done
done

echo "Submitted 210 independent CV jobs (15 folds x 7 DD types x 2 methods)."
echo "Aggregate after completion with aggregate_cv_results.py."
