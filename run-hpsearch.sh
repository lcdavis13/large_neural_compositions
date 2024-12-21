#!/bin/bash

# -------------------------
# 1. User Config: Toggle between parallel or single-job execution of folds
# -------------------------
JOB_NAME="cnode-hpsearch"
TIME_LIMIT="0-00:01:00"  # format is DD-HH:MM:SS
MEMORY="4G"
RUN_FOLDS_IN_PARALLEL=false  # Set to 'true' to run each fold in its own job, 'false' to run all 5 folds in one job
NUM_FOLDS=5                 # Number of folds

# -------------------------
# 2. Define Hyperparameters
# -------------------------
declare -A hyperparams
hyperparams["lr"]="0.001 0.01"
hyperparams["reptile_rate"]="0.0032"
hyperparams["noise"]="0.0 0.025 0.05"

# -------------------------
# 3. Process hyperparameters
# -------------------------
total_combinations=1
for param in "${!hyperparams[@]}"; do
    values=(${hyperparams[$param]}) # Split into array
    total_combinations=$((total_combinations * ${#values[@]}))
done

if [[ "$RUN_FOLDS_IN_PARALLEL" == true ]]; then
    total_combinations=$((total_combinations * NUM_FOLDS))  # Increase total combinations by number of folds
fi

MAX_TASK_ID=$((total_combinations - 1))
echo "Calculated total combinations: $total_combinations (0 to $MAX_TASK_ID)"

# Pass the hyperparameter list and fold configuration as arguments
hyperparam_arg=""
for param in "${!hyperparams[@]}"; do
    hyperparam_arg+="--hyperparams-${param} '${hyperparams[$param]}' "
done

# -------------------------
# 4. Call SBATCH and pass arguments
# -------------------------

# Call the batch script with dynamic SLURM arguments
echo sbatch --job-name=$JOB_NAME \
       --array=0-${MAX_TASK_ID} \
       --time=$TIME_LIMIT \
       --mem=$MEMORY \
       --partition=kill-shared \
       --gpus-per-node=0 \
       --export=ALL \
       hpsearch.slurm \
       --num-folds $NUM_FOLDS \
       --run-folds-in-parallel $RUN_FOLDS_IN_PARALLEL \
       --job-name $JOB_NAME \
       --time-limit $TIME_LIMIT \
       --memory $MEMORY \
       $hyperparam_arg

sbatch --job-name=$JOB_NAME \
       --array=0-${MAX_TASK_ID} \
       --time=$TIME_LIMIT \
       --mem=$MEMORY \
       --partition=kill-shared \
       --gpus-per-node=0 \
       --export=ALL \
       hpsearch.slurm \
       --num-folds $NUM_FOLDS \
       --run-folds-in-parallel $RUN_FOLDS_IN_PARALLEL \
       --job-name $JOB_NAME \
       --time-limit $TIME_LIMIT \
       --memory $MEMORY \
       $hyperparam_arg
