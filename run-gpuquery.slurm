#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=gpu

#SBATCH --array=0-0                # Array range
#SBATCH --output=gpuquery%A_%a.out    # Output files in job-specific folder
#SBATCH --error=gpuquery%A_%a.err     # Error files in job-specific 

#SBATCH --gpus-per-node=NV-V100-SXM2:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000 ## max amount of memory per node you require
#SBATCH --time=0-00:01:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=cnode
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edufolder

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source activate lnc
python util/gpu_query.py
