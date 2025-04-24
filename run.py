import os

# Set a default value if the environment variable is not specified
# os.environ.setdefault("SOLVER", "torchdiffeq")
# os.environ.setdefault("SOLVER", "torchdiffeq_memsafe")
# os.environ.setdefault("SOLVER", "torchode")
# os.environ.setdefault("SOLVER", "torchode_memsafe")
os.environ.setdefault("SOLVER", "trapezoid")

import pandas as pd
import torch

import epoch_managers
import hyperparams
import loss_function
import models
import stream_plot as plotstream
import experiment as expt


def run_experiments(hyperparam_csv=None):
    """
    Run a set of experiments defined by hyperparameters.
    Two modes:
    1. CSV mode: load specific combinations of hyperparameters from a CSV file. 
        - Each row is one complete experiment configuration
    2. Argparse/defaults mode: load hyperparameters from command-line arguments, or use their default values if not specified in the command (if running from IDE without special configs, it will be this version with defaults). 
        - Each argument can be a single value, multiple values to be run in sequential permutations, or a uniform random distribution in linear or exponential scale. 
            - permutations of multiple values: use a comma-separated list e.g. "0.1,0.01,0.001"
            - linearly-uniform random distribution specified as a range with this syntax: "[a...b]" 
            - exponentially-uniform random distribution specified as a range with this syntax: "[a^^^b]" (range must not include zero)
            - UNTESTED: bernoulli random booleans specified with this syntax: "[a]" where a is the probability of True
        - Will run all permutations, so if you have e.g. two parameters with two values and one parameter with three values, it will run 2 * 2 * 3 = 12 experiments.

    Note that any missing arguments will use default values in either mode.
    -permutations/randoms in CSV mode TBD not working yet
    """

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Experiment configuration & hyperparameter space
    # Each is also a command-line argument which can accept multiple comma-separate values for a gridsearch which will be evaluated in sequence.

    hpbuilder, data_param_cat, config_param_cat = hyperparams.construct_hyperparam_composer(hyperparam_csv=hyperparam_csv)

    model_constructors = models.get_model_constructors()

    epoch_mngr_constructors = epoch_managers.get_epoch_manager_constructors()

    loss_fn, score_fn, distr_error_fn = loss_function.get_loss_functions()
    
    all_rows = []

    # loop through possible combinations of dataset hyperparams, though if we aren't in CSV mode there should only be one configuration
    for cp in hpbuilder.parse_and_generate_combinations(category=config_param_cat): 
    
        expt.process_config_params(cp)

        plotstream.set_plot_mode(cp.plot_mode, wait_on_exit=cp.plots_wait_for_exit)

        # loop through possible combinations of dataset hyperparams
        for dp in hpbuilder.parse_and_generate_combinations(category=data_param_cat):

            data_folded, testdata, dense_columns, sparse_columns = expt.process_data_params(dp)

            identity_loss, identity_score = expt.test_identity_model(dp, data_folded, device, loss_fn, score_fn, distr_error_fn)

            # loop through possible combinations of generic hyperparams
            for hp in hpbuilder.parse_and_generate_combinations():

                expt.run_experiment(cp, dp, hp, data_folded, testdata, device, model_constructors, epoch_mngr_constructors, loss_fn, score_fn, distr_error_fn, identity_loss, identity_score, dense_columns, sparse_columns)


    print("\n\nDONE")
    plotstream.finish_up()


# main
if __name__ == "__main__":
    run_experiments()
