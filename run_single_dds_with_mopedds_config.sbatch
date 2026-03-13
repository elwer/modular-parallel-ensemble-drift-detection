#!/bin/bash

#SBATCH --job-name=MOPEDDS_single_members
##SBATCH --cpu-freq=2000000
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8192
##SBATCH --exclusive

cd /data/horse/ws/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection-main
source setup.sh
#python main.py True True False ForestCovertype 1600 HoeffdingTreeClassifier MOPEDDS recent_samples_size 2424 deployment_type threads config_path /data/horse/ws/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection-main/detectors/mopedds/configs/mopedds_forestcover.config
python run_config_detectors.py ForestCovertype $PWD/detectors/mopedds/configs/mopedds.config 2424 1600 True True False
