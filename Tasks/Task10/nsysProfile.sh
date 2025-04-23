#!/bin/bash
#BSUB -J profileCuPyOptimizedFinal	
#BSUB -q gpuv100	
#BSUB -n 5
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -W 10:00
### -- put email for notification -- 	
#BSUB -u s242715@dtu.dk
### -- send notification at completion -- 
#BSUB -N 
### -- Output file (stdout) -- 
#BSUB -o profileCuPy_optimized_final_%J.out		
### -- Output file (stderr) -- 
#BSUB -e profileCuPy_optimized_final_%J.err		

source /dtu/projects/02613_2025/conda/conda_init.sh	
conda activate 02613	

nsys profile -o CuPy_20_profile_optimized_final python CuPy_20_optimized.py 20
