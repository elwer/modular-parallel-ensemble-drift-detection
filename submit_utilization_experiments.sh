#!/bin/bash

for CORES in 1 2 4; do
    echo "Submitting job for ${CORES} core(s)..."
    sbatch \
        --job-name="CPUUtil_${CORES}c" \
        --cpus-per-task=${CORES} \
        --output="utilization_experiments/${CORES}_cores/slurm_%j.out" \
        --export=ALL,CORES=${CORES} \
        run_utilization_experiments.sbatch
done
