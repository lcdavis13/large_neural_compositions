import os

# Set a default value if the environment variable is not specified
# os.environ.setdefault("SOLVER", "torchdiffeq")
# os.environ.setdefault("SOLVER", "torchdiffeq_memsafe")
# os.environ.setdefault("SOLVER", "torchode")
# os.environ.setdefault("SOLVER", "torchode_memsafe")
os.environ.setdefault("SOLVER", "trapezoid")

import torch

import epoch_managers
import hyperparams
import loss_function
import models
import stream_plot as plotstream
import experiment as expt



def main():
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Experiment configuration & hyperparameter space
    # Each is also a command-line argument which can accept multiple comma-separate values for a gridsearch which will be evaluated in sequence.

    hpbuilder, data_param_cat, config_param_cat = hyperparams.construct_hyperparam_composer()

    model_constructors = models.get_model_constructors()

    epoch_mngr_constructors = epoch_managers.get_epoch_manager_constructors()

    loss_fn, score_fn, distr_error_fn = loss_function.get_loss_functions()
    

    cp = hpbuilder.parse_and_generate_combinations(category=config_param_cat)[0] # There should only be one configuration
    
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
    main()
