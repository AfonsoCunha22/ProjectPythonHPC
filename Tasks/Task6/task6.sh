#!/bin/bash
#BSUB -J task_6
#BSUB -q hpc
#BSUB -W 01:00
#BSUB -R "rusage[mem=1024MB]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonE5_2660v3]"
#BSUB -o batch_outputs/task6_%J.out
#BSUB -e batch_outputs/task6_%J.err

# InitializePythonenvironment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Pythonscript
time python task6.py 75