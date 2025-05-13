import argparse
import run

# define arg for CSV file path, mini-epoch size, and data subset
parser = argparse.ArgumentParser(description="Run experiments with different data subsets")
parser.add_argument("--csv_file", type=str, default="batch/HPsearch_256-random_1k_lowEpoch.csv", help="Path to the CSV file containing hyperparameters.")
parser.add_argument("--mini_epoch_size", type=int, default=800, help="Size of mini-epoch for training.")
parser.add_argument("--data_subset", type=int, default=80, help="Data subset size")
parser.add_argument("--data_validation_samples", type=int, default=1000, help="Number of validation samples to reserve. If positive, reserved validation samples are used instead of K-Folds.")

# batchid, taskid, jobid
parser.add_argument("--batchid", type=int, default=0, help="Slurm array job id.")
parser.add_argument("--taskid", type=int, default=0, help="Slurm array task id.")
parser.add_argument("--jobid", type=str, default="-1", help="Slurm job id.")

parser.add_argument("--plot_mode", type=str, default="window", help="Plotting mode: window, inline, or off.")
parser.add_argument("--plots_wait_for_exit", action="store_true", help="Wait for exit after plotting.")

args = parser.parse_args()

overrides = {
    "mini_epoch_size": args.mini_epoch_size, 
    "data_subset": args.data_subset, 
    "data_validation_samples": args.data_validation_samples, 
    "subset_increases_epochs": False, 
    "run_test": True, 
    "eval_benchmarks": False, 
    "reeval_training_set_final": True, 
}

# jobid etc added if not default
if args.batchid != 0:
    overrides["batchid"] = args.batchid
if args.taskid != 0:
    overrides["taskid"] = args.taskid
if args.jobid != "-1":
    overrides["jobid"] = args.jobid
if args.plot_mode != "window":
    overrides["plot_mode"] = args.plot_mode
if args.plots_wait_for_exit:
    overrides["plots_wait_for_exit"] = args.plots_wait_for_exit

if __name__ == "__main__":
    run.run_experiments(hyperparam_csv=args.csv_file, overrides=overrides) 
