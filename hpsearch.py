import itertools
import os
import subprocess

# User Config: Toggle between parallel or single-job execution of folds
NUM_FOLDS = 5  # Number of folds

# Define Hyperparameters
hyperparams = {
    "ode_steps": ["15"],
    "lr": ["0.0001", "0.0032", "0.1"],
    "reptile_rate": ["0.05", "0.5", "1.0"],
    "noise": ["0.0", "0.075"],
    "wd_factor": ["0.0", "2.0"],
    "interpolate": ["0"],
    "num_heads": ["2", "8"],
    "hidden_dim": ["8"],
    "attend_dim_per_head": ["4", "6"],
    "depth": ["4", "16"],
    "ffn_dim_multiplier": ["4"],
    "dropout": ["0.1", "0.5", "0.8"]
}

# Calculate all combinations of hyperparameters
param_names = list(hyperparams.keys())
param_values = list(hyperparams.values())
combinations = list(itertools.product(*param_values))

print(f"Total number of hyperparam combinations: {len(combinations)}")

# Run each combination
for combo_index, combo in enumerate(combinations):
    # Extract values for the current combination
    hyperparam_values = dict(zip(param_names, combo))
    
    # Build the command
    command = [
        "python", "run.py",
        "--kfolds", str(NUM_FOLDS),
        "--headless",
        "--jobid", str(combo_index),
        "--whichfold", "-1",
        "--lr", hyperparam_values["lr"],
        "--reptile_rate", hyperparam_values["reptile_rate"],
        "--mb", "20",
        "--noise", hyperparam_values["noise"],
        "--interpolate", hyperparam_values["interpolate"],
        "--num_heads", hyperparam_values["num_heads"],
        "--hidden_dim", hyperparam_values["hidden_dim"],
        "--attend_dim", hyperparam_values["attend_dim_per_head"],
        "--depth", hyperparam_values["depth"],
        "--ffn_dim_multiplier", hyperparam_values["ffn_dim_multiplier"],
        "--dropout", hyperparam_values["dropout"],
        "--ode_steps", hyperparam_values["ode_steps"],
        "--wd_factor", hyperparam_values["wd_factor"]
    ]
    
    # Print the command for debugging
    print("Command:", " ".join(command))
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        continue
