import subprocess
import csv
import argparse
import sys


parser = argparse.ArgumentParser(description="Re-run hyperparameters from file")
parser.add_argument("--inline", action="store_true", help="Run plots in inline mode, for e.g. notebooks.")
parser.add_argument("--filepath", type=str, default="batch/HPsearch_256-random_1k_lowEpoch.csv", help="Path to the CSV file containing hyperparameters.")


# Path to your configuration CSV file
csv_file = parser.parse_args().filepath
inline = parser.parse_args().inline


# These parameters are flags (no value), include them only if truthy
flag_params = {
    "subset_increases_epochs",
    "run_test",
    "use_best_model",
    "preeval_training_set",
    "reeval_training_set_epoch",
    "reeval_training_set_final",
    "plots_wait_for_exit",
}

# override columns
override = {
    "run_test": "TRUE",
    # "data_validation_samples": 1000,  # If this is positive, reserved validation samples are used instead of K-Folds
    "whichfold": 0, # If this is non-negative, specific fold; if negative, all folds are ran
    "plots_wait_for_exit": "FALSE",
    "plot_mode": "inline" if inline else "window",
}

skip = {
    # "reptile_lr",
    # "headless", 
}

# Load configurations from the CSV file
configurations = []
with open(csv_file, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        configurations.append(row)

# override columns
for key, value in override.items():
    found = False
    for config in configurations:
        if key in config:
            config[key] = value
            found = True
    if not found:
        for config in configurations:
            config[key] = value

# remove columns
for key in skip:
    for config in configurations:
        if key in config:
            del config[key]


def get_valid_args(script_path="run.py"):
    help_cmd = [sys.executable, script_path, "--help"]
    result = subprocess.run(help_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout + result.stderr
    valid_args = set()
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("--"):
            parts = line.split()
            arg = parts[0]
            valid_args.add(arg.lstrip("-"))
    return valid_args


print(f"Total number of configurations: {len(configurations)}")

valid_args = get_valid_args("run.py")

for combo_index, hyperparam_values in enumerate(configurations):
    command = [sys.executable, "run.py"]

    for param_name, param_value in hyperparam_values.items():
        if param_name not in valid_args:
            continue  # Skip unknown params

        if param_name in flag_params:
            if str(param_value).strip().lower() in {"1", "true", "yes"}:
                command.append(f"--{param_name}")
        else:
            command.extend([f"--{param_name}", str(param_value)])

    print("Command:", " ".join(command))
    try:
        result = subprocess.run(
            command,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        continue
