#!/bin/bash
#BSUB -J task_7
#BSUB -q hpc
#BSUB -W 00:30
#BSUB -R "rusage[mem=1024MB]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonE5_2660v3]"
#BSUB -o batch_outputs/task7_%J.out
#BSUB -e batch_outputs/task7_%J.err

# InitializePythonenvironment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Pythonscript
time python task7_jit.py 75