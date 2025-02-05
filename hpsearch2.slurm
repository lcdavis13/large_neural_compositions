#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --job-name=cnode-hpsearch
#SBATCH --array=0-863  ## Make sure this is enough for all hyperparam/fold combos

#SBATCH --time=0-00:10:00  ## time format is DD-HH:MM:SS, 3day max on kill-shared
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
##SBATCH --gpus-per-node=NV-V100-SXM2:1
##SBATCH --gpus-per-node=NV-L40:1
#SBATCH --mem=8G

#SBATCH --output=slurm_out/%A/%A_%a.out
#SBATCH --error=slurm_out/%A/%A_%a.err
#SBATCH --cpus-per-task=1
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edu

echo "This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME"

# -------------------------
# 1. User Config: Toggle between parallel or single-job execution of folds
# -------------------------
RUN_FOLDS_IN_PARALLEL=false  # Set to 'true' to run each fold in its own job, 'false' to run all 5 folds in one job
NUM_FOLDS=5                 # Number of folds

# -------------------------
# 2. Define Hyperparameters
# -------------------------
declare -A hyperparams
hyperparams["ode_steps"]="15"
hyperparams["lr"]="0.0001 0.0032 0.1"
hyperparams["reptile_rate"]="0.05 0.5 1.0"
hyperparams["noise"]="0.0 0.075"
hyperparams["wd_factor"]="0.0 2.0"
hyperparams["interpolate"]="0"
hyperparams["num_heads"]="2 8"
hyperparams["hidden_dim"]="8"
hyperparams["attend_dim_per_head"]="4 6"
hyperparams["depth"]="4 16"
hyperparams["ffn_dim_multiplier"]="4"
hyperparams["dropout"]="0.1 0.5 0.8"

# -------------------------
# 3. Calculate Total Combinations
# -------------------------
total_combinations=1
declare -A hyperparam_sizes
for param in "${!hyperparams[@]}"; do
    values=(${hyperparams[$param]}) # Split into array
    hyperparam_sizes[$param]=${#values[@]} # Store size of each hyperparam
    total_combinations=$((total_combinations * ${#values[@]}))
done

if [[ "$RUN_FOLDS_IN_PARALLEL" == true ]]; then
    total_combinations=$((total_combinations * NUM_FOLDS))  # Increase total combinations by number of folds
fi

# Exit if SLURM_ARRAY_TASK_ID exceeds total combinations
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($total_combinations)."
    exit 1
fi

# -------------------------
# 4. Decompose SLURM_ARRAY_TASK_ID
# -------------------------
declare -A hyperparam_indices
current_id=$SLURM_ARRAY_TASK_ID

# If parallel folds are enabled, separate the last digit to represent the fold number
if [[ "$RUN_FOLDS_IN_PARALLEL" == true ]]; then
    fold_index=$((current_id % NUM_FOLDS)) # Get the fold number (0, 1, ..., NUM_FOLDS-1)
    current_id=$((current_id / NUM_FOLDS)) # Reduce current_id to ignore fold portion
else
    fold_index=-1  # If running all folds in one job, set whichfold to -1
fi

# Calculate hyperparameter indices
for param in "${!hyperparams[@]}"; do
    size=${hyperparam_sizes[$param]}
    hyperparam_indices[$param]=$((current_id % size)) # Get current index
    current_id=$((current_id / size)) # Reduce task_id for the next hyperparam
done

# -------------------------
# 5. Extract Values for Each Hyperparameter
# -------------------------
declare -A hyperparam_values
for param in "${!hyperparams[@]}"; do
    values=(${hyperparams[$param]})
    index=${hyperparam_indices[$param]}
    hyperparam_values[$param]=${values[$index]}
done

# Print debug information
echo "Fold index: $fold_index"
for param in "${!hyperparam_values[@]}"; do
    echo "$param: ${hyperparam_values[$param]}"
done

# -------------------------
# 6. Run the Python Script
# -------------------------
source activate lnc
python run.py --kfolds $NUM_FOLDS --headless \
    --jobid $SLURM_JOB_ID \
    --whichfold $fold_index \
    --lr ${hyperparam_values["lr"]} \
    --reptile_rate ${hyperparam_values["reptile_rate"]} \
    --mb 20 \
    --noise ${hyperparam_values["noise"]} \
    --interpolate ${hyperparam_values["interpolate"]} \
    --num_heads ${hyperparam_values["num_heads"]} \
    --hidden_dim ${hyperparam_values["hidden_dim"]} \
    --attend_dim ${hyperparam_values["attend_dim_per_head"]} \
    --depth ${hyperparam_values["depth"]} \
    --ffn_dim_multiplier ${hyperparam_values["ffn_dim_multiplier"]} \
    --dropout ${hyperparam_values["dropout"]} \
    --ode_steps ${hyperparam_values["ode_steps"]} \
    --wd_factor ${hyperparam_values["wd_factor"]}

