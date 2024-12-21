#!/bin/bash


#SBATCH --output=slurm_out/%A/%A_%a.out    # Output files in job-specific folder
#SBATCH --error=slurm_out/%A/%A_%a.err     # Error files in job-specific folder

##SBATCH --gpus-per-node=1
##SBATCH --gpus-per-node=NV-V100-SXM2:1
##SBATCH --gpus-per-node=NV-L40:1
#SBATCH --cpus-per-task=1

##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edu

echo "Unparsed arguments:"



# -------------------------
# 1. Parse Input Arguments
# -------------------------
while [[ "$#" -gt 0 ]]; do
    echo Param:
    echo $1
    echo Value:
    echo $2
    case $1 in
        --num-folds) NUM_FOLDS="$2"; shift ;;
        --run-folds-in-parallel) RUN_FOLDS_IN_PARALLEL="$2"; shift ;;
        --job-name) JOB_NAME="$2"; shift ;;
        --time-limit) TIME_LIMIT="$2"; shift ;;
        --memory) MEMORY="$2"; shift ;;
        --hyperparams-*)
            param_name="${1/--hyperparams-/}"
            param_name="${param_name//-/_}"  # Replace hyphens with underscores
            declare -A hyperparams
            hyperparams["$param_name"]="$2"
            shift ;;
    esac
    shift
done

# Print debug information
echo "Job Name: $JOB_NAME"
echo "Time Limit: $TIME_LIMIT"
echo "Memory: $MEMORY"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Folds: $NUM_FOLDS"
echo "Run Folds in Parallel: $RUN_FOLDS_IN_PARALLEL"

# -------------------------
# 2. Calculate Total Combinations
# -------------------------
declare -A hyperparam_sizes
total_combinations=1
for param in "${!hyperparams[@]}"; do
    values=(${hyperparams[$param]})
    hyperparam_sizes[$param]=${#values[@]}  # Number of values for each hyperparam
    total_combinations=$((total_combinations * ${#values[@]}))
done

if [[ "$RUN_FOLDS_IN_PARALLEL" == "true" ]]; then
    total_combinations=$((total_combinations * NUM_FOLDS))
fi

# -------------------------
# 3. Decompose Task ID
# -------------------------
current_id=$SLURM_ARRAY_TASK_ID
if [[ "$RUN_FOLDS_IN_PARALLEL" == "true" ]]; then
    fold_index=$((current_id % NUM_FOLDS))
    current_id=$((current_id / NUM_FOLDS))
else
    fold_index=-1
fi

# Calculate hyperparameter indices
declare -A hyperparam_indices
for param in "${!hyperparams[@]}"; do
    size=${hyperparam_sizes[$param]}
    hyperparam_indices[$param]=$((current_id % size))
    current_id=$((current_id / size))
done

# Extract hyperparameter values
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
# 4. Run the Python Script
# -------------------------
source activate lnc
echo run.py --kfolds $NUM_FOLDS --headless \
    --jobid $SLURM_JOB_ID \
    --whichfold $fold_index \
    --lr ${hyperparam_values["lr"]} \
    --reptile_rate ${hyperparam_values["reptile_rate"]} \
    --noise ${hyperparam_values["noise"]}
