#cd /data/quokka/ws/9/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection
cd /data/horse/ws/s4122485-parallelDD/modular-parallel-ensemble-drift-detection
module load release/25.06 GCC/14.2.0 GCCcore/14.2.0 Python
export PATH=/data/horse/ws/s4122485-parallelDD/python3.13/bin:$PATH
source venv/bin/activate
export PYTHON_GIL=0
# python main.py True True False Electricity 1600 HoeffdingTreeClassifier MOPEDDS recent_samples_size 2424 deployment_type threads config_path /data/horse/ws/s4122485-parallelDD/parallel-unsupervised-concept-drift-detection-main/detectors/mopedds/configs/mopedds.config
