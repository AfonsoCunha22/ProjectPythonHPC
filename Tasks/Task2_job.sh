#!/bin/bash

#BSUB -q hpc
#BSUB -J simulation_job
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1024MB]"
#BSUB -M 1GB
#BSUB -W 10
#BSUB -R "select[model == XeonE5_2660v3]"
#BSUB -o simulation_output.log
#BSUB -e simulation_error.log

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

echo "Running simulation for 20 floorplans"
time python simulate.py 20