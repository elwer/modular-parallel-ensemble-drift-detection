#!/bin/bash
#SBATCH --job-name=scalability_study
#SBATCH --output=results_scalability/scalability_%j.log
#SBATCH --error=results_scalability/scalability_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G

# Load environment
module load python 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

pip install --quiet scorep-dummy 2>/dev/null || true

python run_scalability_study.py \
    --stream-length 2000 \
    --drift-frequency 500 \
    --n-repeats 5 \
    --seed 42 \
    --output-dir results_scalability
