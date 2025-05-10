import os
import sys
import torch

os.environ.setdefault("SOLVER", "trapezoid")

import experiment as expt
import hyperparams
import loss_function
import stream_plot as plotstream


def override_dict(target_dict, override_dict):
    for key in target_dict:
        if key in override_dict:
            target_dict[key] = override_dict[key]
    # return target_dict


def run_benchmark_experiments(cli_args=None, hyperparam_csv=None, overrides={}):


    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    hpbuilder, data_param_cat, config_param_cat = hyperparams.construct_hyperparam_composer(hyperparam_csv=hyperparam_csv, cli_args=cli_args)

    loss_fn, score_fns = loss_function.get_loss_functions()

    # loop through possible combinations of dconfig hyperparams, though if we aren't in CSV mode there should only be one configuration
    for cp in hpbuilder.parse_and_generate_combinations(category=config_param_cat): 
        override_dict(cp, overrides)

        expt.process_config_params(cp)

        plotstream.set_plot_mode(cp.plot_mode, wait_on_exit=cp.plots_wait_for_exit)

        # loop through possible combinations of dataset hyperparams
        for dp in hpbuilder.parse_and_generate_combinations(category=data_param_cat):
            override_dict(dp, overrides)

            data_folded, testdata, dense_columns, sparse_columns = expt.process_data_params(dp)

            benchmark_losses = expt.run_benchmarks(cp, dp, data_folded, testdata, score_fns, dense_columns)


    print("\n\nDONE")
    plotstream.finish_up()


# main
if __name__ == "__main__":
    hyperparam_csv = None
    overrides = {}
    
    run_benchmark_experiments(cli_args=sys.argv[1:], hyperparam_csv=hyperparam_csv, overrides=overrides) # capture command line arguments, needs to be done explicitly so that when run_experiments is called from other contexts, CLI args aren't accidentally intercepted 
