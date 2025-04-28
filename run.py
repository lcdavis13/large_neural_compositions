import os
import torch

# Set a default value if the environment variable is not specified (must be done before importing models)
# os.environ.setdefault("SOLVER", "torchdiffeq")
# os.environ.setdefault("SOLVER", "torchdiffeq_memsafe")
# os.environ.setdefault("SOLVER", "torchode")
# os.environ.setdefault("SOLVER", "torchode_memsafe")
os.environ.setdefault("SOLVER", "trapezoid")

import models
import experiment as expt
import epoch_managers
import hyperparams
import loss_function
import stream_plot as plotstream


def override_dict(target_dict, override_dict):
    for key in target_dict:
        if key in override_dict:
            target_dict[key] = override_dict[key]
    # return target_dict


def run_experiments(hyperparam_csv=None, overrides={}):
    """
    Run an experiment or set of experiments, defined by hyperparameters.

    Hyperparameters are determined by the following sources, in order of precedence:
    1. CSV file (inaccessible from command line, passed as an argument to this function)
        - Each row is a separate "base" configuration (but see about permutations below) which will be run sequentially
    2. Command-line arguments
        - Only a single "base" configuration (but see about permutations below)
    3. Default values defined in hyperparams.py

    Each hyperparameter argument can be
     (a) a single literal value
     (b) multiple literal values to be run in sequential permutations, using a comma-separated string e.g. "0.1,0.01,0.001"
        - all list-style arguments will be combinatorically permuted with all other list-style arguments, so if you have e.g. two parameters with two values and one parameter with three values, it will multiply the number of experiments by 2*2*3 = 12
     (c) a uniform random distribution specified as a range with this syntax: "[a...b]" 
     (d) an exponentially-uniform random distribution specified as a range with this syntax: "[a^^^b]" (range must not include zero)
    """

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    hpbuilder, data_param_cat, config_param_cat = hyperparams.construct_hyperparam_composer(hyperparam_csv=hyperparam_csv)

    model_constructors = models.get_model_constructors()

    epoch_mngr_constructors = epoch_managers.get_epoch_manager_constructors()

    loss_fn, score_fn, distr_error_fn = loss_function.get_loss_functions()

    # loop through possible combinations of dconfig hyperparams, though if we aren't in CSV mode there should only be one configuration
    for cp in hpbuilder.parse_and_generate_combinations(category=config_param_cat): 
        override_dict(cp, overrides)

        expt.process_config_params(cp)

        plotstream.set_plot_mode(cp.plot_mode, wait_on_exit=cp.plots_wait_for_exit)

        # loop through possible combinations of dataset hyperparams
        for dp in hpbuilder.parse_and_generate_combinations(category=data_param_cat):
            override_dict(dp, overrides)

            data_folded, testdata, dense_columns, sparse_columns = expt.process_data_params(dp)

            identity_loss, identity_score = expt.test_identity_model(dp, data_folded, device, loss_fn, score_fn, distr_error_fn)

            # loop through possible combinations of generic hyperparams
            for hp in hpbuilder.parse_and_generate_combinations():
                override_dict(hp, overrides)

                expt.run_experiment(cp, dp, hp, data_folded, testdata, device, model_constructors, epoch_mngr_constructors, loss_fn, score_fn, distr_error_fn, identity_loss, identity_score, dense_columns, sparse_columns)


    print("\n\nDONE")
    plotstream.finish_up()


# main
if __name__ == "__main__":
    run_experiments() 
