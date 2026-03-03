#!/bin/bash

#SBATCH --job-name=CPUUtil
#SBATCH --cpu-freq=2000000
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --hint=nomultithread

cd /data/quokka/ws/9/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection
source setup.sh
python run_utilization_experiments.py ${CORES} True True False GasSensor 1600 HoeffdingTreeClassifier recent_samples_size 1000
