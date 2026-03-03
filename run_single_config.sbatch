#!/bin/bash

#SBATCH --job-name=RuntimeExp
#SBATCH --cpu-freq=2000000
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --hint=nomultithread

cd /data/quokka/ws/9/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection
source setup.sh
python run_single_config.py ${DETECTOR} ${CONFIG_ID} ${CORES} ${RUN_ID} PokerHand 1600 HoeffdingTreeClassifier recent_samples_size 1000
