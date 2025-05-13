#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --job-name=cnode-hpsearch
#SBATCH --array=0-120  ## Make sure this is enough for all hyperparam/fold combos

#SBATCH --time=0-16:00:00  ## time format is DD-HH:MM:SS, 3day max on kill-shared
#SBATCH --partition=kill-shared
#SBATCH --gpus-per-node=0
##SBATCH --gpus-per-node=NV-V100-SXM2:1
##SBATCH --gpus-per-node=NV-L40:1
#SBATCH --mem=16G

#SBATCH --output=slurm_out/%A/%A_%a.out
#SBATCH --error=slurm_out/%A/%A_%a.err
#SBATCH --cpus-per-task=1
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edu

echo "This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME"


# -------------------------
# 2. Define Hyperparameters
# Use spaces for parameters of different jobs, commas for multiple parameter values to run sequentially on the same job
# -------------------------
declare -A hyperparams
hyperparams["csv_file_suffix"]="baseline-ConstSoftmax baseline-SLPMultSoftmax baseline-cNODE0 baseline-Linear baseline-LinearSoftmax baseline-SLPSoftmax cNODE1 transformSoftmax cNODE-hourglass canODE-attendFit canODE-FitMat"
hyperparams["mini_epoch_size"]="800"
hyperparams["data_subset"]="1 3 10 32 100 316 1000 3162 10000 31623 100000"
hyperparams["plot_mode"]="off"
hyperparams["data_validation_samples"]="1000"

HPResults_folder="1k"

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

# Exit if SLURM_ARRAY_TASK_ID exceeds total combinations
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($total_combinations)."
    exit 1
fi

# -------------------------
# 4. Decompose SLURM_ARRAY_TASK_ID
# -------------------------
declare -A hyperparam_indices
current_id=$((SLURM_ARRAY_TASK_ID % total_combinations))

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
for param in "${!hyperparam_values[@]}"; do
    echo "$param: ${hyperparam_values[$param]}"
done

echo "Launching job array task: Job ID $SLURM_ARRAY_JOB_ID, Task ID $SLURM_ARRAY_TASK_ID"

# -------------------------
# 6. Run the Python Script
# -------------------------
source activate lnc
python run-datascale.py \
    --batchid $SLURM_ARRAY_JOB_ID \
    --taskid $SLURM_ARRAY_TASK_ID \
    --jobid $SLURM_JOB_ID \
    --csv_file batch/${HPResults_folder}/HPResults_${hyperparam_values["csv_file_suffix"]}.csv \
    --mini_epoch_size ${hyperparam_values["mini_epoch_size"]} \
    --data_subset ${hyperparam_values["data_subset"]} \
    --plot_mode ${hyperparam_values["plot_mode"]} \
    --data_validation_samples ${hyperparam_values["data_validation_samples"]} \

