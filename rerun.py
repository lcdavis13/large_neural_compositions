import run

# hyperparam_csv = "batch/1kLow/HPResults.csv"
# hyperparam_csv = "batch/HPsearch_1k_lowEpoch_Reformat.csv"
# hyperparam_csv = "batch/expanded_cNODE.csv"
hyperparam_csv = "batch/transformer_v_poptrans.csv"
# hyperparam_csv = "batch/1kLow/HPResults_cNODE1.csv"
# hyperparam_csv = "batch/1k/HPResults_baseline-Linear.csv"
overrides = {
    "plot_mode": "window", 
    "plots_wait_for_exit": True, 
    # "reeval_training_set_epoch": True,
    # "whichfold": 0,
    "eval_benchmarks": True,
    # # "data_subset": 1,
    "data_validation_samples": 1000,
    # # "lr": 0.0005,
} 

if __name__ == "__main__":
    run.run_experiments(hyperparam_csv=hyperparam_csv, overrides=overrides) 
