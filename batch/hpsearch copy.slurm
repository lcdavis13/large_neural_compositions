#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --job-name=cnode-hpsearch
#SBATCH --array=0-4607  ## Make sure this is enough for all hyperparam/fold combos

#SBATCH --time=0-16:00:00  ## time format is DD-HH:MM:SS, 3day max on kill-shared
#SBATCH --partition=kill-shared
#SBATCH --gpus-per-node=1
##SBATCH --gpus-per-node=NV-V100-SXM2:1
##SBATCH --gpus-per-node=NV-L40:1
#SBATCH --mem=16G

#SBATCH --output=slurm_out/%A/%A_%a.out
#SBATCH --error=slurm_out/%A/%A_%a.err
#SBATCH --cpus-per-task=1
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edu

echo "This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID , Fold: $fold_index , running on $HOSTNAME"



# -------------------------
# 1. User Config: Toggle between parallel or single-job execution of folds
# -------------------------
RUN_FOLDS_IN_PARALLEL=false  # Set to 'true' to run each fold in its own job, 'false' to run all 5 folds in one job
NUM_FOLDS=5                 # Number of folds


# -------------------------
# 2. Define Hyperparameters
# Use spaces for parameters of different jobs, commas for multiple parameter values to run sequentially on the same job
# -------------------------
declare -A hyperparams
hyperparams["x_dataset"]="256"
hyperparams["y_dataset"]="256-random"
hyperparams["data_subset"]="1000"
hyperparams["data_validation_samples"]="0"
hyperparams["mini_epoch_size"]="0"
hyperparams["model_name"]="baseline-ConstSoftmax,baseline-SLPMultSoftmax,baseline-cNODE0 cNODE1 cNODE-hourglass canODE-FitMat canODE-attendFit transformSoftmax"
hyperparams["epochs"]="[10^^^100] [100^^^1000]"
hyperparams["minibatch_examples"]="250"
hyperparams["ode_timesteps_file"]="t.csv t_shortlinear.csv"
hyperparams["lr"]="[0.000001^^^1.0]"
hyperparams["noise"]="0.0 [0.0001^^^1.0]"
hyperparams["wd"]="0.0 [0.001^^^100.0]"
hyperparams["interpolate"]="False"
hyperparams["interpolate_noise"]="False"
hyperparams["cnode_bias"]="False True"
hyperparams["cnode1_init_zero"]="False True"
hyperparams["identity_gate"]="False True"
hyperparams["num_heads"]="[1...16]"
hyperparams["hidden_dim"]="[8^^^64]"
hyperparams["attend_dim_per_head"]="[1...16]"
hyperparams["depth"]="[2...6]"
hyperparams["ffn_dim_multiplier"]="[0.5^^^4.0]"
hyperparams["dropout"]="0.0 [0.0...0.8]"

# hyperparams["noise"]="0.0 0.033 0.075"
# hyperparams["wd_factor"]="0.0 1.0 2.0 4.0"


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

# # Exit if SLURM_ARRAY_TASK_ID exceeds total combinations
# if [[ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]]; then
#     echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($total_combinations)."
#     exit 1
# fi


# -------------------------
# 4. Decompose SLURM_ARRAY_TASK_ID
# -------------------------
declare -A hyperparam_indices
current_id=$((SLURM_ARRAY_TASK_ID % total_combinations))


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

echo "Launching job array task: Job ID $SLURM_ARRAY_JOB_ID, Task ID $SLURM_ARRAY_TASK_ID, Fold: $fold_index"


# -------------------------
# 6. Run the Python Script
# -------------------------
source activate lnc
python run.py \
    --headless \
    --run_test \
    --reeval_training_set_final \
    --kfolds $NUM_FOLDS \
    --batchid $SLURM_ARRAY_JOB_ID \
    --taskid $SLURM_ARRAY_TASK_ID \
    --jobid $SLURM_JOB_ID \
    --whichfold $fold_index \
    --x_dataset ${hyperparam_values["x_dataset"]} \
    --y_dataset ${hyperparam_values["y_dataset"]} \
    --data_subset ${hyperparam_values["data_subset"]} \
    --data_validation_samples ${hyperparam_values["data_validation_samples"]} \
    --mini_epoch_size ${hyperparam_values["mini_epoch_size"]} \
    --model_name ${hyperparam_values["model_name"]} \
    --epochs ${hyperparam_values["epochs"]} \
    --lr ${hyperparam_values["lr"]} \
    --wd ${hyperparam_values["wd"]} \
    --noise ${hyperparam_values["noise"]} \
    --interpolate ${hyperparam_values["interpolate"]} \
    --interpolate_noise ${hyperparam_values["interpolate_noise"]} \
    --cnode_bias ${hyperparam_values["cnode_bias"]} \
    --cnode1_init_zero ${hyperparam_values["cnode1_init_zero"]} \
    --identity_gate ${hyperparam_values["identity_gate"]} \
    --num_heads ${hyperparam_values["num_heads"]} \
    --hidden_dim ${hyperparam_values["hidden_dim"]} \
    --attend_dim_per_head ${hyperparam_values["attend_dim_per_head"]} \
    --depth ${hyperparam_values["depth"]} \
    --ffn_dim_multiplier ${hyperparam_values["ffn_dim_multiplier"]} \
    --dropout ${hyperparam_values["dropout"]} \
    --ode_timesteps_file ${hyperparam_values["ode_timesteps_file"]} \
    --minibatch_examples ${hyperparam_values["minibatch_examples"]} \

#    --subset_increases_epochs \
