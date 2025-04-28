import argparse
import run

# define arg for CSV file path, mini-epoch size, and data subset
parser = argparse.ArgumentParser(description="Run experiments with different data subsets")
parser.add_argument("--csv_file", type=str, default="batch/HPsearch_256-random_1k_lowEpoch.csv", help="Path to the CSV file containing hyperparameters.")
parser.add_argument("--mini_epoch_size", type=int, default=800, help="Size of mini-epoch for training.")
parser.add_argument("--data_subset", type=str, default=80, help="Comma-separated list of data subset sizes.")
parser.add_argument("--data_validation_samples", type=int, default=1000, help="Number of validation samples to reserve. If positive, reserved validation samples are used instead of K-Folds.")

# batchid, taskid, jobid
parser.add_argument("--batchid", type=int, default=0, help="Slurm array job id.")
parser.add_argument("--taskid", type=int, default=0, help="Slurm array task id.")
parser.add_argument("--jobid", type=str, default="-1", help="Slurm job id.")

args = parser.parse_args()

overrides = {
    "mini_epoch_size": args.mini_epoch_size, 
    "data_subset": args.data_subset, 
    "data_validation_samples": args.data_validation_samples, 
    "subset_increases_epochs": False, 
    "whichfold": 0, 
    "plot_mode": "window", 
    "plots_wait_for_exit": True,
}

# jobid etc added if not default
if args.batchid != 0:
    overrides["batchid"] = args.batchid
if args.taskid != 0:
    overrides["taskid"] = args.taskid
if args.jobid != "-1":
    overrides["jobid"] = args.jobid

if __name__ == "__main__":
    run.run_experiments(args.csv_file, overrides=overrides) 
