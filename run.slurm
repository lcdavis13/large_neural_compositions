#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=gpu

#SBATCH --array=0-4                # Array range
#SBATCH --output=slurm_%A/%A_%a.out    # Output files in job-specific folder
#SBATCH --error=slurm_%A/%A_%a.err     # Error files in job-specific

##SBATCH --nodelist=gpu-0008
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4600 ## max amount of memory per node you require
#SBATCH --time=0-06:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=cnode
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edufolder

# # make directory for results
# mkdir -p slurm_${SLURM_ARRAY_JOB_ID}


echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source activate lnc
python run.py --kfolds 5 --headless --jobid $SLURM_ARRAY_JOB_ID --whichfold $SLURM_ARRAY_TASK_ID
