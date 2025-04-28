from hyperparam_composer import HyperparameterComposer


def construct_hyperparam_composer(hyperparam_csv=None):
    hpbuilder = HyperparameterComposer(hyperparam_csv=hyperparam_csv)

    hpbuilder.add_param("model_name", 
                        # "junk", 
                        # 'baseline-ConstSoftmax',
                        # 'baseline-SLPMultSoftmax',
                        # 'cNODE1',
                        # 'cNODE2',
                        # 'transformSoftmax',
                        # 'transformShaped-AbundEncoding',
                        # 'transformRZShaped',
                        # 'canODE-FitMat',
                        # 'canODE-attendFit',
                        # "canODE-FitMat-AbundEncoding", 
                        'cNODE-hourglass',
                        # 'baseline-cNODE0',
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
                        "256-random",
                        category=datacat, help="dataset to use for supervising outputs")
    hpbuilder.add_param("data_subset", 
                        # 1000,
                        # 10000, 
                        100, 
                        # 100000, 
                        # 1, 10, 100, 1000, 10000, #100000, 
                        category=datacat, help="number of data samples to use, -1 for all")
    hpbuilder.add_param("kfolds", 5, 
                        category=datacat, help="how many data folds, -1 for leave-one-out. If data_validation_samples is <= 0, K-Fold cross-validation will be used. The total samples will be determined by data_subset and divided into folds for training and validation.")
    hpbuilder.add_param("whichfold", -1, 
                        category=datacat, help="which fold to run, -1 for all")
    hpbuilder.add_param("data_validation_samples", 100,
                        category=datacat, help="Number of samples to use for validation. If <= 0, uses K-Fold crossvalidation (see other arguments). If positive, K-Fold will not be used, and instead the first data_validation_samples samples will be used for validation and the following data_subset samples will be used for training.")
    hpbuilder.add_param("minibatch_examples", 100, 
                        help="minibatch size",
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
    hpbuilder.add_flag("plots_wait_for_exit", True,
                        help="wait for plots to be closed by user before exiting",
                        category=config_cat)
    
    # experiment params
    hpbuilder.add_param("epochs", 
                        # 6, 20, 64, 200, 
                        # 64, 
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
                        100, 
                        # 0, 
                        help="number of training samples before running validation and/or tests. If <= 0, uses a full epoch before validation (equivalent to setting mini_epoch_size to the total number of training samples). Default -1.")

    hpbuilder.add_param("early_stop", False, 
                        help="whether or not to use early stopping")
    hpbuilder.add_param("patience", 5, 
                        help="patience for early stopping")
    
    
    # Optimizer params
    hpbuilder.add_param("lr", 
                        # 0.1, 
                        0.00160707665, 
                        # 0.001,
                        # 0.01, 
                        # 1.0, 0.32, 0.1, 0.032, 
                        # 0.01, 0.0032, 0.001, 0.00032, 0.0001, 
                        # 0.32, 0.1, 0.032, 0.01, 0.0032,
                        # 0.32, 0.1, 0.032, 0.01, 0.0032, #0.001, 0.00032,   
                        help="learning rate")
    # hpbuilder.add_param("reptile_lr", 1.0, 
    #                     help="reptile outer-loop learning rate")
    hpbuilder.add_param("wd", 0.0, 
                        help="weight decay")
    hpbuilder.add_param("noise", 
                        0.0,
                        # 0.01,
                        0.032,
                        # 0.1,
                        help="noise level")
    
    # Data augmentation params
    # hpbuilder.add_param("ode_timesteps", 15, 
    #                     help="number of ODE timesteps")
    hpbuilder.add_param("ode_timesteps_file", 
                        # "t.csv",
                        # "t_linear.csv",
                        "t_shortlinear.csv",
                        help="ODE integration timesteps file")
    hpbuilder.add_param("interpolate", False, 
                        help="whether or not to use supervised interpolation steps")
    hpbuilder.add_param("interpolate_noise", False,
                        help="whether or not to use independent noise for interpolation")
    
    # Model architecture params
    hpbuilder.add_param("identity_gate", 
                        True,
                        # False, 
                        help="whether or not to use 'ReZero'-style learnable gate scalars, initialized such that each model starts as an identity function")
    hpbuilder.add_param("cnode1_init_zero", 
                        True, 
                        # False, 
                        help="whether or not to use 'ReZero'-style gates to ensure all models start as an identity function")
    hpbuilder.add_param("cnode_bias", 
                        True, 
                        # False,
                        help="whether or not to use a bias term when predicting fitness in cNODE and similar models")
    hpbuilder.add_param("num_heads", 15,  
                        help="number of attention heads in transformer-based models")
    hpbuilder.add_param("hidden_dim", 8, 
                        help="hidden dimension")
    hpbuilder.add_param("attend_dim_per_head", 9,
                        help="dimension of attention embedding, per attention head")
    hpbuilder.add_param("depth", 2, 
                        help="depth of model")
    hpbuilder.add_param("ffn_dim_multiplier", 
                        # 4.0,
                        3.924555754,  
                        help="multiplier for feedforward network dimension in transformer-based models")
    hpbuilder.add_param("dropout", 0.04137076975, 
                        help="dropout rate")
                        
    return hpbuilder, datacat, config_cat

