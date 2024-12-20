#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --job-name=cnode-hpsearch
#SBATCH --array=0-99  ## Make sure this is enough for all hyperparam/fold combos

#SBATCH --time=0-00:30:00  ## time format is DD-HH:MM:SS, 3day max on kill-shared
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
##SBATCH --gpus-per-node=NV-V100-SXM2:1
##SBATCH --gpus-per-node=NV-L40:1
#SBATCH --mem=4G

#SBATCH --output=slurm_out/%A/%A_%a.out
#SBATCH --error=slurm_out/%A/%A_%a.err
#SBATCH --cpus-per-task=1
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edu

echo "This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME"

# -------------------------
# 1. Define Hyperparameters
# -------------------------
declare -A hyperparams
hyperparams["lr"]="0.001 0.01 0.1 1.0 10.0"
hyperparams["reptile_rate"]="0.0032 0.032 0.1 0.5 1.0"
hyperparams["noise"]="0.025 0.05 0.075 0.1"


# -------------------------
# 2. Calculate Total Combinations
# -------------------------
total_combinations=1
declare -A hyperparam_sizes
for param in "${!hyperparams[@]}"; do
    values=(${hyperparams[$param]}) # Split into array
    hyperparam_sizes[$param]=${#values[@]} # Store size of each hyperparam
    total_combinations=$((total_combinations * ${#values[@]}))
done

# Exit if SLURM_ARRAY_TASK_ID exceeds total combinations
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID exceeds total combinations ($total_combinations)."
    exit 1
fi

# -------------------------
# 3. Decompose SLURM_ARRAY_TASK_ID
# -------------------------
declare -A hyperparam_indices
current_id=$SLURM_ARRAY_TASK_ID

for param in "${!hyperparams[@]}"; do
    size=${hyperparam_sizes[$param]}
    hyperparam_indices[$param]=$((current_id % size)) # Get current index
    current_id=$((current_id / size)) # Reduce task_id for the next hyperparam
done

# -------------------------
# 4. Extract Values for Each Hyperparameter
# -------------------------
declare -A hyperparam_values
for param in "${!hyperparams[@]}"; do
    values=(${hyperparams[$param]})
    index=${hyperparam_indices[$param]}
    hyperparam_values[$param]=${values[$index]}
done

# Print debug information
for param in "${!hyperparam_values[@]}"; do
    echo "$param: ${hyperparam_values[$param]}"
done

# -------------------------
# 5. Run the Python Script
# -------------------------
source activate lnc
python run.py --kfolds 5 --headless \
    --jobid $SLURM_JOB_ID \
    --whichfold -1 \
    --lr ${hyperparam_values["lr"]} \
    --reptile_rate ${hyperparam_values["reptile_rate"]} \
    --mb 20 \
    --wd 0.0 \
    --ode_steps=30 \
    --noise ${hyperparam_values["noise"]}

