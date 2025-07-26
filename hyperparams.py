from hyperparam_composer import HyperparameterComposer


def construct_hyperparam_composer(hyperparam_csv=None, cli_args=None):
    hpbuilder = HyperparameterComposer(hyperparam_csv=hyperparam_csv, cli_args=cli_args)

    hpbuilder.add_param("model_name", 
                        # "junk", 
                        # 'cNODE1', 
                        # 'cNODE1+1-proper',
                        # 'cNODE1+1-colFrozen',
                        # 'cNODE1+1-noFreeze',
                        # 'cNODE2',
                        # 'cNODE2Improved',
                        # 'EmbeddedSimplexIdentity',
                        # 'EmbeddedReplicatorIdentity',
                        # 'Constant',
                        # 'SimplexConstant',
                        # 'ReplicatorConstant',
                        # 'Linear',
                        # 'SimplexLinear',
                        # 'ReplicatorLinear',
                        # 'ShallowMLP',
                        # 'SimplexShallowMLP',
                        # 'ReplicatorShallowMLP',
                        # 'ShallowMLP2',
                        # 'SimplexShallowMLP2',
                        # 'ReplicatorShallowMLP2',
                        # 'ResidualMLP',
                        # 'SimplexResidualMLP',
                        # 'ReplicatorResidualMLP',
                        # 'SimplexTransformer',
                        'ReplicatorTransformer',
                        'SimplexPopTransformer',
                        'ReplicatorPopTransformer',
                        help="model(s) to run")

    # data params
    datacat = "data"
    # hpbuilder.add_param("dataset", 
    #                     # "waimea", 
    #                     # "waimea-std", 
    #                     # "cNODE-paper-ocean", 
    #                     # "cNODE-paper-ocean-std", 
    #                     # "cNODE-paper-human-oral", 
    #                     # "cNODE-paper-human-oral-std", 
    #                     "69@4_48_richness50",
    #                     # "5000@7_48_richness170",
    #                     category=datacat, help="dataset to use")
    hpbuilder.add_param("x_dataset", 
                        "256",
                        category=datacat, help="dataset to use for inputs")
    hpbuilder.add_param("y_dataset", 
                        # "random-2-old",
                        # "random-0-gLV",
                        "random-1-gLV",
                        category=datacat, help="dataset to use for supervising outputs")
    hpbuilder.add_param("data_subset", 
                        800,  # 1k HP search
                        # 80000,  # 100k HP search
                        # 1000,
                        # 3162,
                        # 10000, 
                        # 100, 
                        # 100000, 
                        # 1, 10, 100, 1000, 10000, #100000, 
                        category=datacat, help="number of data samples to use, -1 for all")
    hpbuilder.add_param("kfolds", 5, 
                        category=datacat, help="how many data folds, -1 for leave-one-out. If data_validation_samples is <= 0, K-Fold cross-validation will be used. The total samples will be determined by data_subset and divided into folds for training and validation.")
    hpbuilder.add_param("whichfold", 
                        # -1,
                        0,  
                        category=datacat, help="which fold to run, -1 for all")
    hpbuilder.add_param("data_validation_samples", 100,
                        category=datacat, help="Number of samples to use for validation. If <= 0, uses K-Fold crossvalidation (see other arguments). If positive, K-Fold will not be used, and instead the first data_validation_samples samples will be used for validation and the following data_subset samples will be used for training.")
    hpbuilder.add_param("minibatch_examples", 
                        250, 
                        # 100, 
                        help="minibatch size",
                        category=datacat)
    hpbuilder.add_flag("eval_benchmarks", True,
                        help="whether or not to evaluate the benchmark models on the dataset", 
                        category=datacat) 
    
    # slurm params
    config_cat = "config"
    hpbuilder.add_param("batchid", 0,
                        help="slurm array job id", 
                        category=config_cat)
    hpbuilder.add_param("taskid", 0,
                        help="slurm array task id", 
                        category=config_cat)
    hpbuilder.add_param("jobid", "-1", 
                        help="slurm job id", 
                        category=config_cat)
    hpbuilder.add_param("plot_mode", 
                        "window",
                        # "inline",
                        # "off",
                        help="plotting mode: window, inline, or off", 
                        category=config_cat)
    hpbuilder.add_flag("plots_wait_for_exit", 
                        # False,
                        True,
                        help="wait for plots to be closed by user before exiting",
                        category=config_cat)
    
    # experiment params
    hpbuilder.add_param("epochs",
                        # 182,  # 1k HP search
                        # 7,  # 100k HP search
                        # 300, # extra epochs for randomized timesteps
                        # 2.0,  
                        # 6, 20, 64, 200, 
                        # 12.0, 
                        # 64.0, 
                        25.0, 
                        # 300, 
                        # 200, 
                        help="maximum number of epochs")
    hpbuilder.add_flag("subset_increases_epochs", 
                        # True,
                        False, 
                        help="if true, epochs will be adjusted based on the subset size to run the same number of total samples")
    hpbuilder.add_param("base_data_subset", -1,
                        help="Base data subset size to use for calculating epochs when subset_increases_epochs is true. If -1, the max dataset size is used instead.",)
    hpbuilder.add_param("min_epochs", 1, 
                        help="minimum number of epochs")
    hpbuilder.add_param("accumulated_minibatches", 1, 
                        help="number of minibatches to accumulate before stepping")
    hpbuilder.add_flag("run_test", 
                        True,
                        # False,
                        category=datacat, help="run the test set after training")
    hpbuilder.add_flag("use_best_model", 
                        # True,
                        False,
                        help="whether or not to use the best model for testing")
    hpbuilder.add_flag("export_cnode",
                        # False,
                        True,
                        help="whether or not to export the cNODE parameters after training")
    hpbuilder.add_flag("preeval_training_set", False,
                        help="whether or not to pre-evaluate the training set before starting training")
    hpbuilder.add_flag("reeval_training_set_epoch", False,
                        help="whether or not to re-evaluate the training set after each epoch")
    hpbuilder.add_flag("reeval_training_set_final", True,
                        help="whether or not to re-evaluate the training set after the final epoch")
    
    hpbuilder.add_param("epoch_manager", 
                        "Fixed", 
                        # "AdaptiveValPlateau",
                        help="which type of epoch manager to use")
    hpbuilder.add_param("mini_epoch_size", 
                        # 500,
                        # 100, 
                        0, 
                        help="number of training samples before running validation and/or tests. If <= 0, uses a full epoch before validation (equivalent to setting mini_epoch_size to the total number of training samples). Default -1.")

    hpbuilder.add_param("early_stop", False, 
                        help="whether or not to use early stopping")
    hpbuilder.add_param("patience", 5, 
                        help="patience for early stopping")
    
    
    # Optimizer params
    hpbuilder.add_param("lr", 
                        # 0.003993407529,  # 1k HP search
                        # 0.001223800286,  # 100k HP search
                        # 0.00003993407529,  # randomized timesteps
                        # 0.1, 
                        # 0.00160707665, 
                        # 0.001,
                        0.01, 
                        # 1.0, 0.32, 0.1, 0.032, 
                        # 1.0, 0.1, 0.01, 0.001, 
                        # 0.0001, 0.00001, 0.000001,
                        # 0.32, 0.1, 0.032, 0.01, 0.0032,
                        # 0.32, 0.1, 0.032, 0.01, 0.0032, #0.001, 0.00032, 
                        # 0.032, 0.01, 0.0032, 0.001,   
                        help="learning rate")
    # hpbuilder.add_param("reptile_lr", 1.0, 
    #                     help="reptile outer-loop learning rate")
    hpbuilder.add_param("wd", 
                        0.0, 
                        # 0.0822,  
                        # 2.034777678,  # 1k HP search
                        # 0.8170174678,  # 100k HP search
                        help="weight decay")
    hpbuilder.add_param("noise", 
                        0.0,
                        # 0.01,
                        # 0.032,
                        # 0.1,
                        # 0.5,
                        help="noise level")
    
    # Data augmentation params
    # hpbuilder.add_param("ode_timesteps", 15, 
    #                     help="number of ODE timesteps")
    hpbuilder.add_param("ode_timesteps_file", 
                        # "t.csv",
                        # "t_linear.csv",
                        "t_shortlinear.csv",
                        help="ODE integration timesteps file")
    hpbuilder.add_param("number_converged_timesteps", 
                        # 1, 
                        # 29, 
                        15, 
                        help="number of timesteps in the timesteps file that were after convergence to a stable fixed point. If > 1, multiple timesteps will be used for training, encouraging the model to learn a stable fixed point.")
    hpbuilder.add_param("interpolate", False, 
                        help="whether or not to use supervised interpolation steps")
    hpbuilder.add_param("interpolate_noise", False,
                        help="whether or not to use independent noise for interpolation")
    
    # Model architecture params
    hpbuilder.add_param("env_dim", 
                        # 0, 
                        1, 
                        # 26,
                        help="Number of 'environment' species added to envNode. If 0, the model becomes cNODE1.")
    hpbuilder.add_param("env_scale", 
                        # 0.00001,
                        # 0.0001,
                        # 0.001,
                        # 0.01,
                        1.0/72.0, 
                        # 0.1,
                        # 1.0,
                        # 10.0,
                        # 72.0,
                        # 100.0,
                        help="Initial abundance value of 'environment' species in glvNode/envNode. A reasonable default is 1/mean_richness. However, for some unfathomable reason, glv1NODE seems to function better with approximately mean_richness instead of the inverse.")
    hpbuilder.add_param("hidden_dim", 512, 
                        help="hidden dimension for MLP-based models")
    hpbuilder.add_param("embed_dim", 12,
                        help="dimension of embedding for embedding-based models")
    hpbuilder.add_param("num_blocks", 2, 
                        help="depth of model")
    hpbuilder.add_param("learnable_skip", 
                        True, 
                        # False, 
                        help="Whether or not to use learnable scalar weights on the skip connections.")
    hpbuilder.add_param("dropout", 
                        0.1, 
                        help="dropout rate")
    hpbuilder.add_param("num_heads", 4, 
                        help="number of attention heads in transformer-based models")
    hpbuilder.add_param("fcn_dim_factor", 
                        1.0, 
                        help="dim multiplier for FCN sublayer in transformer blocks, relative to embedding dimension")
    hpbuilder.add_param("attn_dropout", 
                        0.1, 
                        help="dropout rate in attention layers")
    hpbuilder.add_param("fcn_dropout", 
                        0.1, 
                        help="dropout rate for FCN sublayer in transformer blocks")
    hpbuilder.add_param("enrich_blocks", 0, 
                        help="number of transformer blocks to use for enriching embeddings in embedding-based ODE models")
    hpbuilder.add_param("fitness_blocks", 2, 
                        help="number of transformer blocks to use for the fitness function in embedding-based ODE models")
    hpbuilder.add_param("pop_block_depth", 2, 
                        help="number of population blocks to use in explicit (non-ODE) population-weighted transformer. This is roughly analogous to the number of ODE integration steps.")
    hpbuilder.add_param("use_logx", 
                        False, 
                        # True, 
                        help="whether to use normed_log(x) instead of x as input for x-encoding in embedded replicator models.")

    # Model architecture re-parameterization params
    hpbuilder.add_param("parameter_target", 
                        0, 
                        # 256 ** 2, 
                        # 2_000_000, 
                        help="target number of parameters for the model. If <= 0, the model will not be re-parameterized and will use only explicit architecture parameters.")
    hpbuilder.add_param("width_depth_tradeoff",
                        # 0.0, 
                        # 0.25, 
                        0.5, 
                        # 0.75,
                        # 1.0,  
                        help="tradeoff between width and depth for re-parameterization when parameter_target > 0. 0.0 means all width, 1.0 means all depth, and 0.5 is a balanced tradeoff.")

    return hpbuilder, datacat, config_cat


if __name__ == "__main__":
    # run run.py
    from run import run_experiments
    run_experiments(hyperparam_csv=None, cli_args=None, overrides={})