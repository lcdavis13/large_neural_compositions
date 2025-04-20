import os
import subprocess
import csv

# User Config: Set number of folds
NUM_FOLDS = 5

# Path to your configuration CSV file
CSV_FILE = "batch/HPsearch_256-random_1k.csv"

# These parameters are flags (no value), include them only if truthy
FLAG_PARAMS = {
    "headless",
    "subset_increases_epochs",
    "run_test",
    "use_best_model",
    "preeval_training_set",
    "reeval_training_set_epoch",
    "reeval_training_set_final"
}

# override columns
override = {
    "headless": "FALSE",
    "run_test": "TRUE",
    "subset_increases_epochs": "TRUE",
}

# Load configurations from the CSV file
configurations = []
with open(CSV_FILE, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        configurations.append(row)

# override columns
for key, value in override.items():
    for config in configurations:
        if key in config:
            config[key] = value

print(f"Total number of configurations: {len(configurations)}")

# Run each configuration from the configurations table
for combo_index, hyperparam_values in enumerate(configurations):
    # Start building the command
    command = [
        "python", "run.py",
    ]

    # Add parameters from the configuration table
    for param_name, param_value in hyperparam_values.items():
        if param_name in FLAG_PARAMS:
            if str(param_value).strip().lower() in {"1", "true", "yes"}:
                command.append(f"--{param_name}")
        else:
            command.extend([f"--{param_name}", str(param_value)])

    # Print the command for debugging
    print("Command:", " ".join(command))

    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        continue
