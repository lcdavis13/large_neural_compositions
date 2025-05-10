import run

# hyperparam_csv = "batch/1kLow/HPResults.csv"
hyperparam_csv = "batch/1kLow/HPResults_cNODE1.csv"
overrides = {
    "plot_mode": "window", 
    "plots_wait_for_exit": True, 
    "reeval_training_set_epoch": True,
    "whichfold": 0,
    "eval_benchmarks": True,
    } 

if __name__ == "__main__":
    run.run_experiments(hyperparam_csv=hyperparam_csv, overrides=overrides) 
