#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=gpu

# Calculate the total number of combinations:
# 5 (mb values) x 5 (lr values) x 5 (reptile_rate values) = 125
#SBATCH --array=0-124                # Array range for 125 total combinations (0 to 124)
#SBATCH --output=slurm_out/%A/%A_%a.out    # Output files in job-specific folder
#SBATCH --error=slurm_out/%A/%A_%a.err     # Error files in job-specific folder

#SBATCH --gpus-per-node=1
##SBATCH --gpus-per-node=NV-V100-SXM2:1
##SBATCH --gpus-per-node=NV-L40:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G ## max amount of memory per node you require
#SBATCH --time=0-00:30:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=cnode-hpsearch
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edu

echo "This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME"

# Define the arrays of values for learning rate (lr), reptile_rate, and mb
lr_values=(0.0001 0.001 0.01 0.1 1.0)
reptile_rate_values=(0.0032 0.032 0.1 0.5 1.0)
mb_values=(176 64 32 20 10)

# Calculate the indices for the lr, reptile_rate, and mb from SLURM_ARRAY_TASK_ID
# Total combinations = number of mb values * number of lr values * number of reptile_rate values

total_lr_values=${#lr_values[@]}  # 5
reptile_rate_total=${#reptile_rate_values[@]}  # 5
mb_total=${#mb_values[@]}  # 5

total_combinations=$((mb_total * total_lr_values * reptile_rate_total))  # 5 x 5 x 5 = 125

# Decompose SLURM_ARRAY_TASK_ID into mb, lr, and reptile_rate indices
mb_index=$((SLURM_ARRAY_TASK_ID % mb_total))  # Which mb to run (0 to 4)
lr_index=$(( (SLURM_ARRAY_TASK_ID / mb_total) % total_lr_values ))  # Which lr to use (0 to 4)
reptile_rate_index=$(( (SLURM_ARRAY_TASK_ID / (mb_total * total_lr_values)) % reptile_rate_total ))  # Which reptile_rate to use (0 to 4)

# Get the specific values for mb, lr, and reptile_rate
mb=${mb_values[$mb_index]}
lr=${lr_values[$lr_index]}
reptile_rate=${reptile_rate_values[$reptile_rate_index]}

echo "MB index: $mb_index"
echo "Mini-batch size (mb): $mb"
echo "Learning rate (lr): $lr"
echo "Reptile rate: $reptile_rate"

# Load python environment, then call the python script with calculated arguments
source activate lnc
python run.py --kfolds 5 --headless --jobid $SLURM_JOB_ID --whichfold -1 --lr $lr --reptile_rate $reptile_rate --mb $mb
