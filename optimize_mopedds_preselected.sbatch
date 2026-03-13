#!/bin/bash

#SBATCH --job-name=DD_MOPEDDS_PreSel
#SBATCH --cpu-freq=2000000
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=4096
##SBATCH --exclusive

cd /data/quokka/ws/9/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection
source setup.sh
python optimization/mopedds_preselected_optimize.py --n_trials "${N_TRIALS}" --dataset "${DATASET}"
