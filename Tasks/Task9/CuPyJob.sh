#!/bin/bash
### -- Job name --	
#BSUB -J CuPy_all
### -- Queue name --		
#BSUB -q gpuv100	
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- ask for number of nodes(default: 1) -- 
#BSUB -R "span[hosts=1]"
### -- Resources(mem) -- 
#BSUB -R "rusage[mem=6GB]"
### -- GPU model -- 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- Resources(mem) -- 
#BSUB -R "select[gpu32gb]"
### -- Wall-clocktime -- 
#BSUB -W 08:00
### -- put email for notification -- 	
#BSUB -u s242715@dtu.dk
### -- send notification at completion -- 
#BSUB -N 
### -- Output file (stdout) -- 
#BSUB -o cupy_all_%J.out		
### -- Output file (stderr) -- 
#BSUB -e cupy_all_%J.err		

source /dtu/projects/02613_2025/conda/conda_init.sh	
conda activate 02613	

### put before commands --  print information on the CPU type 
python CuPy.py 4571